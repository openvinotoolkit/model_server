#ifndef MEDIAPIPE_GRAPHS_BYTETRACK_CALCULATORS_BASETRACK_H_
#define MEDIAPIPE_GRAPHS_BYTETRACK_CALCULATORS_BASETRACK_H_

namespace mediapipe {
namespace bytetrack {

class BaseTrack {
public:
    enum class TrackState { NEW,
        TRACKED,
        LOST,
        REMOVED };

    // mirrors Python's next_id() — static counter owned here
    static int next_id() { return ++count_; }
    static void reset_id() { count_ = 0; }

    // Accessors
    int track_id() const { return track_id_; }
    int frame_id() const { return frame_id_; }
    int start_frame() const { return start_frame_; }
    float score() const { return score_; }
    TrackState state() const { return state_; }
    bool is_activated() const { return is_activated_; }

    void MarkLost() { state_ = TrackState::LOST; }
    void MarkRemoved() { state_ = TrackState::REMOVED; }

protected:
    int track_id_ = 0;
    int frame_id_ = 0;
    int start_frame_ = 0;
    float score_ = 0.f;
    bool is_activated_ = false;
    TrackState state_ = TrackState::NEW;
    static int count_;
};

}  // namespace bytetrack
}  // namespace mediapipe

#endif