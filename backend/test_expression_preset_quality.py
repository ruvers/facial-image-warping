from __future__ import annotations

from backend.liveportrait_bridge import _LivePortraitExpressionRuntime


def test_expression_preset_quality_scores_separate_smile_and_laugh() -> None:
    runtime = object.__new__(_LivePortraitExpressionRuntime)
    baseline = {
        "mouth_width": 0.34,
        "mouth_open": 0.028,
        "corner_raise": 0.010,
        "mouth_asymmetry": 0.004,
        "left_eye_open": 0.055,
        "right_eye_open": 0.055,
        "eye_open": 0.055,
        "eye_asymmetry": 0.0,
        "cheek_y": 0.560,
        "brow_position": 0.350,
        "brow_frown": 0.0,
    }
    smile = {
        **baseline,
        "mouth_width": 0.40,
        "mouth_open": 0.060,
        "corner_raise": 0.050,
        "eye_open": 0.047,
        "cheek_y": 0.535,
    }
    laugh = {
        **baseline,
        "mouth_width": 0.43,
        "mouth_open": 0.112,
        "corner_raise": 0.052,
        "eye_open": 0.036,
        "cheek_y": 0.528,
    }
    shock = {
        **baseline,
        "mouth_width": 0.34,
        "mouth_open": 0.190,
        "corner_raise": -0.010,
        "eye_open": 0.078,
        "mouth_asymmetry": 0.016,
    }

    laugh_score = runtime.score_laugh_frame(laugh, baseline)
    smile_as_laugh_score = runtime.score_laugh_frame(smile, baseline)
    shock_as_laugh_score = runtime.score_laugh_frame(shock, baseline)
    shock_scores = runtime._expression_quality_scores(shock, baseline)

    assert laugh_score > smile_as_laugh_score
    assert laugh_score > shock_as_laugh_score
    assert shock_scores["scream_penalty"] > 1.0
    assert runtime._candidate_rejected("laugh", shock_scores) is True


if __name__ == "__main__":
    test_expression_preset_quality_scores_separate_smile_and_laugh()
    print("expression preset quality ok")
