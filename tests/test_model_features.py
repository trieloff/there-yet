import datetime
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import model_features


class ModelFeaturesGitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        self._run_git('init')
        self._run_git('config', 'user.email', 'test@example.com')
        self._run_git('config', 'user.name', 'Test User')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_git(self, *args, env=None):
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        subprocess.run(
            ['git', *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
            env=merged_env,
        )

    def _commit(self, message, day_index):
        ts = datetime.datetime(2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=day_index)
        env = {
            'GIT_AUTHOR_DATE': ts.isoformat(),
            'GIT_COMMITTER_DATE': ts.isoformat(),
        }
        payload = self.repo / 'payload.txt'
        payload.write_text(f'{message}\n{day_index}\n', encoding='utf-8')
        self._run_git('add', 'payload.txt')
        self._run_git('commit', '-m', message, env=env)

    def _tag(self, name, day_index):
        ts = datetime.datetime(2024, 1, 1, 13, 0, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=day_index)
        env = {'GIT_COMMITTER_DATE': ts.isoformat()}
        self._run_git('tag', '-a', name, '-m', name, env=env)

    def test_load_tag_data_filters_non_semver(self):
        self._commit('chore: init', 0)
        self._tag('v0.1.0', 0)
        self._commit('feat: add one', 1)
        self._tag('release-candidate', 1)
        self._commit('feat: add two', 2)
        self._tag('v0.2.0', 2)

        tags = model_features.load_tag_data(self.repo)
        self.assertEqual(['v0.1.0', 'v0.2.0'], [name for name, _ in tags])

    def test_count_feat_between_tags_counts_only_feat_subjects(self):
        self._commit('chore: init', 0)
        self._tag('v0.1.0', 0)

        self._commit('feat: add base command', 1)
        self._commit('feat(api): add endpoint', 2)
        self._commit('fix: patch bug', 3)
        self._tag('v0.2.0', 3)

        self._commit('feat: add docs command', 4)
        self._commit('docs: update readme', 5)
        self._tag('v0.3.0', 5)

        tags = model_features.load_tag_data(self.repo)
        feat_counts = model_features.count_feat_between_tags(self.repo, tags)

        self.assertEqual([('v0.2.0', 2), ('v0.3.0', 1)], [(name, count) for name, _, count in feat_counts])


class ModelFeaturesReleaseTests(unittest.TestCase):
    def test_parse_release_rows_filters_semver_and_sorts(self):
        raw = '\n'.join(
            [
                'v1.1.0\t2024-06-01T12:00:00Z',
                'draft-tag\t2024-01-01T12:00:00Z',
                'v1.0.0\t2024-01-01T12:00:00Z',
            ]
        )

        rows = model_features.parse_release_rows(raw)
        self.assertEqual(['v1.0.0', 'v1.1.0'], [name for name, _ in rows])

    def test_normalize_remote_repo_name(self):
        self.assertEqual('owner/repo', model_features.normalize_remote_repo_name('owner/repo'))
        self.assertEqual('owner/repo', model_features.normalize_remote_repo_name('https://github.com/owner/repo.git'))


if __name__ == '__main__':
    unittest.main()
