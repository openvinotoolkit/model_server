//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "bytetrack_calculator.h"
#include "src/bytetrack/calculators/bytetrack_calculator.pb.h"

#include <algorithm>
#include <unordered_map>

#include <Eigen/Dense>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "src/bytetrack/calculators/strack.h"
#include "src/bytetrack/calculators/kalman_filter.h"
#include "src/bytetrack/calculators/matching_utils.h"

// node {
//   calculator: "ByteTrackCalculator"
//   input_stream: "DETECTIONS:detections_with_id"
//   output_stream: "DETECTIONS:tracked_detections"
//   options: {
//     [mediapipe.ByteTrackCalculatorOptions.ext] {
//       track_high_threshold:0.5
//       track_low_threshold:0.1
//       new_track_threshold:0.25
//       matching_threshold: 0.75
//       track_buffer: 30
//       fuse_score: true
//     }
//   }
// }

namespace mediapipe {
REGISTER_CALCULATOR(ByteTrackCalculator);

absl::Status ByteTrackCalculator::GetContract(CalculatorContract* cc) {
    cc->Inputs().Get("DETECTIONS", 0).Set<std::vector<Detection>>();
    cc->Outputs().Get("DETECTIONS", 0).Set<std::vector<Detection>>();
    return absl::OkStatus();
}

absl::Status ByteTrackCalculator::Open(CalculatorContext* cc) {
    options_ = cc->Options<mediapipe::ByteTrackCalculatorOptions>();
    match_thresh_ = options_.matching_threshold();
    track_buffer_ = options_.track_buffer();
    track_high_thresh_ = options_.track_high_threshold();
    track_low_thresh_ = options_.track_low_threshold();
    new_track_thresh_ = options_.new_track_threshold();
    fuse_score_ = options_.fuse_score();
    return absl::OkStatus();
}

absl::Status ByteTrackCalculator::Process(CalculatorContext* cc) {
    if (cc->Inputs().Get("DETECTIONS", 0).IsEmpty()) {
        return absl::OkStatus();  // nothing to do
    }
    int64_t current_ts = cc->InputTimestamp().Microseconds();
    if (last_timestamp_ > 0 && frame_id_ > 1) {
        float dt_sec = (current_ts - last_timestamp_) / 1e6f;
        float instant_fps = 1.0f / dt_sec;
        // Smooth it with a running average to avoid jitter
        estimated_fps_ = 0.9f * estimated_fps_ + 0.1f * instant_fps;
        max_time_lost_ = static_cast<int>(estimated_fps_ / 30.0f * track_buffer_);
        LOG(INFO) << "MAX TIME LOST: " << max_time_lost_;
    }
    last_timestamp_ = current_ts;
    frame_id_++;
    LOG(INFO) << "Frame " << frame_id_
              << "  tracked_=" << tracked_stracks_.size()
              << "  lost_=" << lost_stracks_.size();

    LOG(INFO) << "ByteTrackCalculator::Process called, frame " << frame_id_;
    auto high_dets = std::make_unique<std::vector<Detection>>();
    auto low_dets = std::make_unique<std::vector<Detection>>();

    if (!cc->Inputs().Tag("DETECTIONS").IsEmpty()) {
        const auto& input = cc->Inputs().Tag("DETECTIONS").Get<std::vector<Detection>>();
        for (const auto& detection : input) {
            float score = detection.score(0);
            if (score >= track_high_thresh_)
                high_dets->push_back(detection);
            else if (score >= track_low_thresh_)
                low_dets->push_back(detection);
        }
    }

    std::vector<bytetrack::STrack> activated_stracks;
    std::vector<bytetrack::STrack> refind_stracks;
    std::vector<bytetrack::STrack> lost_stracks;
    std::vector<bytetrack::STrack> removed_stracks;

    LOG(INFO) << "High dets: " << high_dets->size()
              << " Low dets: " << low_dets->size();

    // create tracks from high score detections
    std::vector<bytetrack::STrack> detections;
    if (high_dets->size() > 0) {
        for (auto& d : *high_dets) {
            detections.emplace_back(d);
        }
    }

    // add newly detected tracks
    std::vector<bytetrack::STrack*> unconfirmed;
    std::vector<bytetrack::STrack*> tracked_stracks;
    for (auto& track : tracked_stracks_) {
        if (!track.is_activated()) {
            unconfirmed.push_back(&track);
        } else {
            tracked_stracks.push_back(&track);
        }
    }

    /////////////////////// First association //////////////////////////////////////
    std::vector<bytetrack::STrack*> lost_ptrs;
    for (auto& t : lost_stracks_)
        lost_ptrs.push_back(&t);

    auto strack_pool = JointStracks(tracked_stracks, lost_ptrs);
    bytetrack::STrack::MultiPredict(strack_pool);
    auto dists = BuildIoUCostMatrix(strack_pool, detections);
    if (fuse_score_) {
        dists = FuseScore(dists, detections);
    }
    // printDists(dists);
    // gives 3 vectors
    auto [matches, u_track, u_detection] = bytetrack::LinearAssignment(dists, match_thresh_);

    LOG(INFO) << "First step association, "
              << " matches size: " << matches.size()
              << ", U_track size: " << u_track.size()
              << ", U_detection size: " << u_detection.size();

    for (int k = 0; k < matches.rows(); ++k) {
        int itracked = matches(k, 0);
        int idet = matches(k, 1);
        auto* track = strack_pool[itracked];
        auto det = detections[idet];
        if (track->state() == bytetrack::BaseTrack::TrackState::TRACKED) {
            track->Update(detections[idet], frame_id_);
            activated_stracks.push_back(*track);
        } else {
            track->ReActivate(det, frame_id_, false);
            refind_stracks.push_back(*track);
        }
    }
    LOG(INFO) << "  After 1st assoc: activated=" << activated_stracks.size()
              << "  refind=" << refind_stracks.size();
    /////////////////////// Second association ///////////////////////////
    std::vector<bytetrack::STrack> detections_second;
    if (low_dets->size() > 0) {
        for (const auto& d : *low_dets) {
            detections_second.emplace_back(d);
        }
    }

    std::vector<bytetrack::STrack*> r_tracked_stracks;
    for (int i : u_track) {
        if (strack_pool[i]->state() == bytetrack::BaseTrack::TrackState::TRACKED) {
            r_tracked_stracks.push_back(strack_pool[i]);
        }
    }

    dists = BuildIoUCostMatrix(r_tracked_stracks, detections_second);
    if (fuse_score_) {
        dists = FuseScore(dists, detections_second);
    }

    auto [matches2, u_track2, u_detection_second] = bytetrack::LinearAssignment(dists, 0.5f);

    LOG(INFO) << "Second step association, "
              << " matches2 size: " << matches2.size()
              << ", U_track2 size: " << u_track2.size()
              << ", U_detection_second size: " << u_detection_second.size();

    for (int k = 0; k < matches2.rows(); ++k) {
        int itracked = matches2(k, 0);
        int idet = matches2(k, 1);
        auto* track = r_tracked_stracks[itracked];
        auto det = detections_second[idet];
        if (track->state() == bytetrack::BaseTrack::TrackState::TRACKED) {
            track->Update(det, frame_id_);
            activated_stracks.push_back(*track);
        } else {
            track->ReActivate(det, frame_id_, false);
            refind_stracks.push_back(*track);
        }
    }

    // mark lost tracks
    for (int it : u_track2) {
        auto* track = r_tracked_stracks[it];
        if (track->state() != bytetrack::BaseTrack::TrackState::LOST) {
            track->MarkLost();
            lost_stracks.push_back(*track);
        }
    }
    LOG(INFO) << "  After 2nd assoc: lost=" << lost_stracks.size();
    /////////////////////// DEAL W UNCONFIRMED TRACKS ///////////////////////////
    std::vector<bytetrack::STrack> detections_uc;
    for (int i : u_detection) {
        detections_uc.push_back(detections[i]);
    }

    dists = BuildIoUCostMatrix(unconfirmed, detections_uc);
    if (fuse_score_) {
        dists = FuseScore(dists, detections_uc);
    }

    auto [matches3, u_unconfirmed, u_detection_3] = bytetrack::LinearAssignment(dists, 0.7f);

    for (int k = 0; k < matches3.rows(); ++k) {
        int itracked = matches3(k, 0);
        int idet = matches3(k, 1);
        unconfirmed[itracked]->Update(detections_uc[idet], frame_id_);
        activated_stracks.push_back(*unconfirmed[itracked]);
    }

    for (int it : u_unconfirmed) {
        auto* track = unconfirmed[it];
        track->MarkRemoved();
    }
    /////////////////////// INITIALIZE NEW TRACKS /////////////////////////////
    for (int inew : u_detection_3) {
        auto track = detections_uc[inew];
        if (track.score() < new_track_thresh_) {
            continue;
        }
        track.Activate(&kalman_filter_, frame_id_);
        activated_stracks.push_back(track);
    }
    LOG(INFO) << "  After unconfirmed+new: activated=" << activated_stracks.size();
    /////////////////////// UPDATE STATE /////////////////////////////
    for (auto& track : lost_stracks_) {
        LOG(INFO) << "Time diff update state " << frame_id_ - track.frame_id() << ",";
        if (frame_id_ - track.frame_id() > max_time_lost_) {
            track.MarkRemoved();
            removed_stracks.push_back(track);
        }
    }

    LOG(INFO) << "actiavted_stracks size: " << activated_stracks.size()
              << ",refind_stracks size: " << refind_stracks.size()
              << ",lost_stracks size: " << lost_stracks.size()
              << ",removed_stracks size: " << removed_stracks.size();

    // Filter tracked_stracks_ to only TRACKED state, then join activated + refind
    // (mirrors: self.tracked_stracks = [t for t in self.tracked_stracks if t.state == Tracked])
    {
        std::vector<bytetrack::STrack> only_tracked;
        for (auto& t : tracked_stracks_) {
            if (t.state() == bytetrack::BaseTrack::TrackState::TRACKED)
                only_tracked.push_back(t);
        }
        tracked_stracks_ = only_tracked;
    }

    // joint_stracks(tracked_stracks_, activated_stracks)
    {
        std::vector<bytetrack::STrack*> cur_ptrs, act_ptrs, ref_ptrs;
        for (auto& t : tracked_stracks_)
            cur_ptrs.push_back(&t);
        for (auto& t : activated_stracks)
            act_ptrs.push_back(&t);
        for (auto& t : refind_stracks)
            ref_ptrs.push_back(&t);

        auto joined = JointStracks(cur_ptrs, act_ptrs);
        // joint_stracks(tracked_stracks_, refind_stracks)
        joined = JointStracks(joined, ref_ptrs);

        std::vector<bytetrack::STrack> joined_tracks;
        joined_tracks.reserve(joined.size());
        for (auto* t : joined)
            joined_tracks.push_back(*t);
        tracked_stracks_ = std::move(joined_tracks);
    }

    // sub_stracks(lost_stracks_, tracked_stracks_)  then extend with lost_stracks (local)
    {
        std::vector<bytetrack::STrack*> lost_ptrs2, new_tracked_ptrs, removed_ptrs, local_lost_ptrs;
        for (auto& t : lost_stracks_)
            lost_ptrs2.push_back(&t);
        for (auto& t : tracked_stracks_)
            new_tracked_ptrs.push_back(&t);  // fresh pointers!
        for (auto& t : removed_stracks_)
            removed_ptrs.push_back(&t);
        for (auto& t : lost_stracks)
            local_lost_ptrs.push_back(&t);

        auto new_lost = SubStracks(lost_ptrs2, new_tracked_ptrs);
        new_lost = JointStracks(new_lost, local_lost_ptrs);  // extend
        new_lost = SubStracks(new_lost, removed_ptrs);

        // lost_stracks_.clear();
        // for (auto* t : new_lost) lost_stracks_.push_back(*t);
        std::vector<bytetrack::STrack> l_stracks;
        l_stracks.reserve(new_lost.size());
        for (auto* t : new_lost)
            l_stracks.push_back(*t);
        lost_stracks_ = std::move(l_stracks);
    }

    // extend removed_stracks_
    for (auto& t : removed_stracks)
        removed_stracks_.push_back(t);

    // remove duplicates
    {
        std::vector<bytetrack::STrack*> tracked_ptrs2, lost_ptrs3;
        for (auto& t : tracked_stracks_)
            tracked_ptrs2.push_back(&t);
        for (auto& t : lost_stracks_)
            lost_ptrs3.push_back(&t);
        auto [dedup_tracked, dedup_lost] = RemoveDuplicateStracks(tracked_ptrs2, lost_ptrs3);
        std::vector<bytetrack::STrack> new_tracked, new_lost;
        new_tracked.reserve(dedup_tracked.size());
        new_lost.reserve(dedup_lost.size());
        for (auto* t : dedup_tracked)
            new_tracked.push_back(*t);
        for (auto* t : dedup_lost)
            new_lost.push_back(*t);
        tracked_stracks_ = std::move(new_tracked);
        lost_stracks_ = std::move(new_lost);
    }

    LOG(INFO) << "After update state";
    LOG(INFO) << "  End of frame: tracked=" << tracked_stracks_.size()
              << "  lost=" << lost_stracks_.size();

    ////////////////////////////// OUTPUT ///////////////////////////////////
    auto output = std::make_unique<std::vector<Detection>>();
    for (const auto& t : tracked_stracks_) {
        if (!t.is_activated())
            continue;
        Detection d;
        d.set_detection_id(t.track_id());
        d.add_label(t.label());
        d.add_score(t.score());
        auto* loc = d.mutable_location_data();
        loc->set_format(LocationData::RELATIVE_BOUNDING_BOX);
        auto* rb = loc->mutable_relative_bounding_box();
        Eigen::Vector4f box = t.tlwh();
        rb->set_xmin(box(0));
        rb->set_ymin(box(1));
        rb->set_width(box(2));
        rb->set_height(box(3));
        output->push_back(d);
    }
    LOG(INFO) << "After building detections";
    LOG(INFO) << "  Output size: " << output->size();
    // cc->Outputs().Get("DETECTIONS",0).Add(
    // output.release(), cc->InputTimestamp());
    cc->Outputs().Get("DETECTIONS", 0).Add(output.release(), cc->Inputs().Get("DETECTIONS", 0).Value().Timestamp());
    return absl::OkStatus();
}

std::vector<bytetrack::STrack*> ByteTrackCalculator::JointStracks(std::vector<bytetrack::STrack*>& a, std::vector<bytetrack::STrack*>& b) {
    std::unordered_map<int, bool> exists;
    std::vector<bytetrack::STrack*> res;

    for (auto* t : a) {
        exists[t->track_id()] = true;
        res.push_back(t);
    }

    for (auto* t : b) {
        int tid = t->track_id();
        if (exists.find(tid) == exists.end()) {
            exists[tid] = true;
            res.push_back(t);
        }
    }
    return res;
}

std::vector<bytetrack::STrack*> ByteTrackCalculator::SubStracks(std::vector<bytetrack::STrack*>& a, std::vector<bytetrack::STrack*>& b) {
    std::unordered_map<int, bytetrack::STrack*> exists;
    std::vector<bytetrack::STrack*> res;
    for (auto* t : a) {
        exists[t->track_id()] = t;
    }
    for (auto* t : b) {
        int tid = t->track_id();
        if (exists.find(tid) != exists.end()) {
            exists.erase(tid);
        }
    }
    for (auto& i : exists) {
        res.push_back(i.second);
    }
    return res;
}

//// WIP
std::pair<std::vector<bytetrack::STrack*>, std::vector<bytetrack::STrack*>>
ByteTrackCalculator::RemoveDuplicateStracks(std::vector<bytetrack::STrack*>& a, std::vector<bytetrack::STrack*>& b) {

    auto pdist = BuildIoUCostMatrix(a, b);

    std::vector<int> dupa, dupb;
    for (int p = 0; p < (int)a.size(); ++p) {
        for (int q = 0; q < (int)b.size(); ++q) {
            if (pdist(p, q) < 0.15f) {  // high overlap — duplicate
                int timep = a[p]->frame_id() - a[p]->start_frame();
                int timeq = b[q]->frame_id() - b[q]->start_frame();
                if (timep > timeq)
                    dupb.push_back(q);
                else
                    dupa.push_back(p);
            }
        }
    }

    std::vector<bytetrack::STrack*> resa, resb;
    for (int i = 0; i < (int)a.size(); ++i)
        if (std::find(dupa.begin(), dupa.end(), i) == dupa.end())
            resa.push_back(a[i]);
    for (int i = 0; i < (int)b.size(); ++i)
        if (std::find(dupb.begin(), dupb.end(), i) == dupb.end())
            resb.push_back(b[i]);

    return {resa, resb};
}

}  // namespace mediapipe
