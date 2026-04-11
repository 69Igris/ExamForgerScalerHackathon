"""
test_inference_logging.py
=========================
Verifies that inference.py emits EXACTLY the required log format.

This is the most important test for automated evaluation — the evaluator
parses [START], [STEP], [END] lines and will silently give 0 scores if
the format is wrong.

Run: python -m pytest test_inference_logging.py -v
"""
import io
import json
import re
import sys
import pytest

# ─── Log format regex patterns ────────────────────────────────────────────────
START_PATTERN = re.compile(
    r'^\[START\] task=(?P<task>\S+) env=(?P<env>\S+) model=(?P<model>\S+)$'
)
STEP_PATTERN = re.compile(
    r'^\[STEP\] step=(?P<step>\d+) action=(?P<action>.+) '
    r'reward=(?P<reward>-?\d+\.\d{2}) done=(?P<done>true|false) '
    r'error=(?P<error>\S+)$'
)
END_PATTERN = re.compile(
    r'^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) '
    r'score=(?P<score>\d+\.\d{3}) rewards=(?P<rewards>[\d.,\-]+)$'
)


def capture_logs(func, *args, **kwargs):
    """Capture stdout from a function call and return lines."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines, result


class TestLogFormat:
    """Tests that verify exact [START]/[STEP]/[END] log format compliance."""

    def test_log_start_format(self):
        """[START] line must have task=, env=, model= fields."""
        sys.path.insert(0, ".")
        from inference import log_start

        lines, _ = capture_logs(log_start, "question_generation", "examforge", "test-model")
        assert len(lines) == 1, f"log_start must emit exactly 1 line, got {len(lines)}"

        line = lines[0]
        match = START_PATTERN.match(line)
        assert match is not None, (
            f"[START] line does not match required format.\n"
            f"Got:      {line!r}\n"
            f"Expected: [START] task=<name> env=<benchmark> model=<model>"
        )
        assert match.group("task") == "question_generation"
        assert match.group("env") == "examforge"
        assert match.group("model") == "test-model"

    def test_log_step_format_with_json_action(self):
        """[STEP] line must have action= as parseable string, reward to 2dp, done lowercase."""
        sys.path.insert(0, ".")
        from inference import log_step

        action_json = json.dumps({"action_type": "generate_question", "topic": "Kinematics"})
        lines, _ = capture_logs(log_step, 1, action_json, 0.30, False, None)

        assert len(lines) == 1, f"log_step must emit exactly 1 line, got {len(lines)}"
        line = lines[0]
        match = STEP_PATTERN.match(line)
        assert match is not None, (
            f"[STEP] line does not match required format.\n"
            f"Got:      {line!r}\n"
            f"Expected: [STEP] step=N action=... reward=X.XX done=bool error=null"
        )
        assert match.group("step") == "1"
        assert match.group("reward") == "0.30"
        assert match.group("done") == "false"
        assert match.group("error") == "null"
        # Action must be parseable as part of a JSON context
        action_str = match.group("action")
        assert "action_type" in action_str, (
            f"action= field must contain JSON with action_type. Got: {action_str!r}"
        )

    def test_log_step_done_true_lowercase(self):
        """done=true must be lowercase, not True or TRUE."""
        sys.path.insert(0, ".")
        from inference import log_step

        lines, _ = capture_logs(log_step, 5, '{"action_type":"assemble_paper"}', 2.10, True, None)
        line = lines[0]
        assert "done=true" in line, f"done must be lowercase 'true'. Got: {line!r}"
        assert "done=True" not in line, "done must NOT be capitalized"

    def test_log_step_error_field(self):
        """error= must be 'null' when no error, or error string when error."""
        sys.path.insert(0, ".")
        from inference import log_step

        # No error
        lines_no_err, _ = capture_logs(log_step, 1, '{"action_type":"noop"}', 0.0, False, None)
        assert "error=null" in lines_no_err[0], f"No error must show error=null. Got: {lines_no_err[0]}"

        # With error
        lines_err, _ = capture_logs(log_step, 2, '{"action_type":"noop"}', -0.20, False, "Invalid topic")
        assert "error=Invalid" in lines_err[0], f"Error must show in error= field. Got: {lines_err[0]}"

    def test_log_end_format(self):
        """[END] line must have success=, steps=, score=, rewards= all correct."""
        sys.path.insert(0, ".")
        from inference import log_end

        rewards = [0.30, 0.70, 2.10]
        lines, _ = capture_logs(log_end, True, 3, 0.820, rewards)

        assert len(lines) == 1, f"log_end must emit exactly 1 line, got {len(lines)}"
        line = lines[0]
        match = END_PATTERN.match(line)
        assert match is not None, (
            f"[END] line does not match required format.\n"
            f"Got:      {line!r}\n"
            f"Expected: [END] success=bool steps=N score=X.XXX rewards=r1,r2,rN"
        )
        assert match.group("success") == "true"
        assert match.group("steps") == "3"
        assert match.group("score") == "0.820"
        assert match.group("rewards") == "0.30,0.70,2.10"

    def test_log_end_score_3_decimal_places(self):
        """score= must be formatted to exactly 3 decimal places."""
        sys.path.insert(0, ".")
        from inference import log_end

        lines, _ = capture_logs(log_end, False, 10, 0.5, [0.5])
        line = lines[0]
        assert "score=0.500" in line, f"Score must be 3 decimal places. Got: {line!r}"

    def test_log_end_rewards_comma_separated_2dp(self):
        """rewards= must be comma-separated, each to 2 decimal places."""
        sys.path.insert(0, ".")
        from inference import log_end

        lines, _ = capture_logs(log_end, True, 5, 0.750, [0.3, 0.7, -0.2, 1.5, 2.1])
        line = lines[0]
        assert "rewards=0.30,0.70,-0.20,1.50,2.10" in line, (
            f"rewards must be comma-separated 2dp floats. Got: {line!r}"
        )

    def test_deterministic_baseline_produces_valid_logs(self):
        """_run_deterministic_episode must emit valid [START], [STEP]s, [END]."""
        sys.path.insert(0, ".")
        from server.environment import SUBJECT_TOPICS
        from inference import _run_deterministic_episode

        lines, score = capture_logs(
            _run_deterministic_episode,
            "question_generation",
            "JEE Physics",
            list(SUBJECT_TOPICS["JEE Physics"])
        )

        # Must have exactly 1 [START]
        start_lines = [l for l in lines if l.startswith("[START]")]
        assert len(start_lines) == 1, f"Must have exactly 1 [START] line. Got {len(start_lines)}"

        # Must have at least 1 [STEP]
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        assert len(step_lines) >= 1, "Must have at least 1 [STEP] line"

        # Must have exactly 1 [END]
        end_lines = [l for l in lines if l.startswith("[END]")]
        assert len(end_lines) == 1, f"Must have exactly 1 [END] line. Got {len(end_lines)}"

        # Score must be valid
        assert isinstance(score, float), "Score must be float"
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"
        assert score > 0.3, f"Deterministic baseline score too low: {score}"

        # All [STEP] lines must have JSON action
        for step_line in step_lines:
            assert "action_type" in step_line, (
                f"[STEP] action must contain action_type. Got: {step_line!r}"
            )
