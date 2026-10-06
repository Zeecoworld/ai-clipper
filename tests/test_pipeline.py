"""Run with: python -m unittest discover tests  (needs ffmpeg + opencv; no Whisper needed)."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autoclip import config, pipeline
from autoclip.captions import CaptionStyle, Word, build_ass, group_words
from autoclip.media import probe
from autoclip.scoring import blend_scores, speech_scores


def make_test_video(path: Path, seconds=12, audio=True):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=25:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


class CaptionTests(unittest.TestCase):
    def test_group_words_breaks_on_sentence_end(self):
        ws = [Word(t, i, i + .5) for i, t in enumerate(["Hi", "there.", "How", "are", "you"])]
        self.assertEqual([[w.text for w in g] for g in group_words(ws, 3)],
                         [["Hi", "there."], ["How", "are", "you"]])

    def test_ass_has_events_and_escapes(self):
        ws = [Word("Hello", 0.0, 0.4), Word("{wor}ld", 0.4, 0.9)]
        ass = build_ass(ws, CaptionStyle(hook_text="Hook!"), 10)
        self.assertIn("PlayResX: 1080", ass)
        self.assertEqual(ass.count("Dialogue:"), 3)  # hook + 2 words
        self.assertNotIn("{wor}", ass)


class SpeechScoreTests(unittest.TestCase):
    def test_hook_dense_speech_outscores_silence(self):
        ws = [Word(t, 10 + i * .3, 10.3 + i * .3) for i, t in enumerate(["why", "is", "this", "the", "biggest", "mistake?"])]
        sc = dict(speech_scores(ws, [0.0, 11.0]))
        self.assertGreater(sc[11.0], sc[0.0])
        self.assertEqual(sc[0.0], 0.0)

    def test_blend_weights(self):
        motion, speech = [(0, 1.0), (1, 0.0)], [(0, 0.0), (1, 1.0)]
        self.assertEqual(blend_scores(motion, speech, 0.0)[0][1], 1.0)
        self.assertEqual(blend_scores(motion, speech, 1.0)[1][1], 1.0)

    def test_pipeline_speech_mode_picks_and_captions(self):
        tmp = Path(tempfile.mkdtemp()); config.WORK_ROOT = tmp / "work"
        try:
            src = tmp / "in.mp4"; make_test_video(src)
            fake = [Word("why", 6.0, 6.3), Word("mistake?", 6.3, 6.9)]
            with mock.patch.object(pipeline, "transcribe", return_value=fake):
                _, res = pipeline.process(src, pipeline.new_job_dir(), 1, 5, 25, True, None,
                                          CaptionStyle(), highlight_mode="speech")
            self.assertEqual(len(res), 1)
            self.assertGreaterEqual(res[0].start, 1.0)
            self.assertEqual(res[0].caption_words, 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        config.WORK_ROOT = self.tmp / "work"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, audio=True, captions=True, hook=""):
        src = self.tmp / "in.mp4"
        make_test_video(src, audio=audio)
        job = pipeline.new_job_dir()
        fake = [Word("this", .2, .5), Word("is", .5, .7), Word("a", .7, .8), Word("test", .8, 1.4)]
        with mock.patch.object(pipeline, "transcribe", return_value=fake):
            return pipeline.process(src, job, 2, 5, 25, captions, None,
                                    CaptionStyle(hook_text=hook))

    def test_end_to_end_with_captions(self):
        info, res = self._run(hook="Watch this")
        self.assertTrue(res)
        for r in res:
            p = probe(r.path)
            self.assertEqual((p.width, p.height), (1080, 1920))
            self.assertTrue(p.has_audio)
            self.assertEqual(r.caption_words, 4)

    def test_no_audio_skips_captions(self):
        _, res = self._run(audio=False)
        self.assertTrue(res)
        self.assertIn("no audio", res[0].caption_note.lower())
        self.assertFalse(probe(res[0].path).has_audio)


if __name__ == "__main__":
    unittest.main()
