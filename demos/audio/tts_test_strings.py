#!/usr/bin/env python3
"""Send a battery of tricky TTS test strings to an OpenAI-compatible
speech endpoint, one by one, and save each result as a numbered WAV file.

Usage:
    python tts_test_strings.py --endpoint http://localhost:8000/v3 \
                               --model kokoro \
                               [--voice None] \
                               [--output-dir tts_output]
"""

import argparse
import os
import sys
import time

from openai import OpenAI

TEST_STRINGS = [
    'Dr. A. B. Carter Jr. met Sen. O\'Neill at 5 p.m., Wed., in Washington, D.C.',
    'Mr. Smith, Ph.D., arrived on Fri. at 6:30 a.m.; Mrs. Jones left at noon.',
    'We meet on 01/02/2025 at 05:30 IST; is that India or Israel time?',
    'The deadline is 2025\u201102\u201101 23:59 UTC\u221205:00 (EST).',
    'He finished 1st; she was 22nd\u2014barely.',
    'Prices: $1,234.56 vs \u20ac1.234,56; also \u00a512 345 (thin space).',
    'Add \u00be cup, then \u00bd tsp; total \u2248 1\u00bc cups.',
    'Chapter XLIV starts on page ix; version v2.0.0 follows v1.12.9.',
    'Dose: 5 mg vs 5 \u03bcg\u2014don\'t confuse micrograms with milligrams.',
    'Avogadro\'s number is 6.022e23; \u03c0 \u2248 3.14159; \u221a2 \u2248 1.4142.',
    'Temperature dropped to \u221210 \u00b0C (14 \u00b0F) with 90% RH.',
    'Visit https://example.com/a/b?x=1&y=2#frag or email ops+alerts@example.org.',
    'Open C:\\Program Files\\Project\\config.yaml or /usr/local/bin/run.sh.',
    '.NET, Node.js, C#, C++17, and Rust\'s crate\u2011names\u2011with\u2011hyphens.',
    '"WYSIWYG," "GIF" (hard or soft g?), "SQL" (sequel or S\u2011Q\u2011L?).',
    'I will present the present to the lead singer who stepped on the lead.',
    'They desert the desert; the dove dove; he wound the wound.',
    'Please record the record before the minute is up in a minute.',
    'She sells seashells by the seashore; truly Irish wristwatch.',
    'Unique New York, toy boat, red leather yellow leather.',
    'A na\u00efve co\u00f6perative fa\u00e7ade in S\u00e3o Paulo; \u0141\u00f3d\u017a and Krak\u00f3w in Poland.',
    'Pi\u00f1ata, jalape\u00f1o, cr\u00e8me br\u00fbl\u00e9e, bouillabaisse, d\u00e9j\u00e0 vu.',
    '\U0001f44d\U0001f3fb is a thumbs\u2011up with light skin tone; \U0001f9d1\u200d\U0001f4bb writes code; \U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466 is a family; \U0001f1f5\U0001f1f1 is a flag.',
    'Faces: \U0001f642\U0001f609\U0001f610\U0001f611\U0001f636; hearts: \u2764\ufe0f\U0001f9e1\U0001f49b\U0001f49a\U0001f499; mixed: \U0001f937\u200d\u2642\ufe0f\U0001f926\u200d\u2640\ufe0f.',
    'Latin "A" vs Cyrillic "\u0410"; Greek "\u03c1" vs Latin "p"; micro "\u00b5" vs Greek "\u03bc".',
    '\u05e9\u05dc\u05d5\u05dd and \u0645\u0631\u062d\u0628\u064b\u0627 appear with left\u2011to\u2011right text in one line.',
    'Prosody markers: \u02c8primary, \u02ccsecondary, and length \u02d0 are tricky for tokenizers.',
    'Arrows for intonation: \u2197 rising, \u2198 falling, \u2193 drop.',
    'He said, "She replied, \'no\u2014never\u2026\'," then left\u2014silently.',
    'Parentheticals (like this\u2014really!) and em\u2011dashes\u2014here\u2014confuse prosody.',
    'Let f(x)=x^2; then d/dx x^2=2x; \u2202/\u2202x is the operator.',
    'Inline code x += 1; and TeX E=mc^2 should be read clearly.',
    'N,N\u2011Diethyl\u2011meta\u2011toluamide (DEET) differs from p\u2011xylene and m\u2011cresol.',
    'The RFC 7231/HTTP\u2011semantics "GET" vs "HEAD" distinction matters.',
    'Read "macOS" vs "Mac OS", "iOS", "SQL", "URL", and "S3" correctly.',
]


def main():
    parser = argparse.ArgumentParser(
        description="Send TTS test strings to an OpenAI-compatible speech endpoint."
    )
    parser.add_argument(
        "--endpoint", required=True,
        help="Base URL of the API (e.g. http://localhost:8000/v3)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name to use for speech generation"
    )
    parser.add_argument(
        "--voice", default=None,
        help="Voice name (default: voice1)"
    )
    parser.add_argument(
        "--output-dir", default="tts_output",
        help="Directory to save output WAV files (default: tts_output)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    client = OpenAI(base_url=args.endpoint, api_key="unused")

    total = len(TEST_STRINGS)
    print(f"Sending {total} test strings to {args.endpoint} (model={args.model}, voice={args.voice})")
    print(f"Output directory: {args.output_dir}\n")

    succeeded = 0
    failed = 0
    total_size_kb = 0.0
    t_start = time.time()

    for idx, text in enumerate(TEST_STRINGS, start=1):
        preview = text[:80] + ("..." if len(text) > 80 else "")
        print(f"[{idx:2d}/{total}] {preview}")

        out_path = os.path.join(args.output_dir, f"{idx:02d}.wav")
        t0 = time.time()
        try:
            response = client.audio.speech.create(
                model=args.model,
                voice=args.voice,
                input=text,
            )
            response.write_to_file(out_path)
            elapsed = time.time() - t0
            size_kb = os.path.getsize(out_path) / 1024
            total_size_kb += size_kb
            succeeded += 1
            print(f"        -> {out_path}  ({size_kb:.1f} KB, {elapsed:.2f}s)")
        except Exception as exc:
            elapsed = time.time() - t0
            failed += 1
            print(f"        !! FAILED after {elapsed:.2f}s: {exc}", file=sys.stderr)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Summary: {succeeded} succeeded, {failed} failed out of {total}")
    print(f"Total time: {total_elapsed:.2f}s  (avg {total_elapsed/total:.2f}s per string)")
    print(f"Total audio size: {total_size_kb:.1f} KB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
