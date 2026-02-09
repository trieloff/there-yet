#!/usr/bin/env python3
"""
Model arrival curves for local or remote repos.

- Counts conventional `feat:` commits between semver tags (vX.Y.Z)
- Or, in GH-only mode, counts semver GitHub Releases over time
- Fits Gompertz, NHPP Goel-Okumoto, and NHPP Weibull models
- Reports fit diagnostics (SSE, RMSE, R2)
- Renders PNG plots using Pillow (no matplotlib dependency)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

SEMVER_TAG_RE = re.compile(r'^v\d+\.\d+\.\d+$')
FEAT_SUBJECT_RE = re.compile(r'^feat(\(.+\))?:')
DEFAULT_HOMEBREW_GH = Path('/opt/homebrew/bin/gh')
GH_BIN = 'gh'


@dataclass
class FitDiagnostics:
    sse: float
    rmse: float
    r2: float


@dataclass
class AnalysisResult:
    source: str
    out_path: Path
    start_date: datetime.date
    end_date: datetime.date
    observed_count: int
    observed_label: str
    point_count: int
    point_label: str
    gompertz_K: float
    goel_okumoto_a: float
    weibull_a: float
    weibull_c: float
    diagnostics: dict[str, FitDiagnostics]


def run_git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ['git', *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        cmd = ' '.join(['git', *args])
        stderr = exc.stderr.strip()
        raise RuntimeError(f"{cmd} failed in {repo}: {stderr}") from exc
    return completed.stdout


def run_gh(*args: str) -> str:
    try:
        completed = subprocess.run(
            [GH_BIN, *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError('gh CLI is not installed') from exc
    except subprocess.CalledProcessError as exc:
        cmd = ' '.join([GH_BIN, *args])
        stderr = exc.stderr.strip()
        raise RuntimeError(f"{cmd} failed: {stderr}") from exc
    return completed.stdout


def is_git_repo(repo: Path) -> bool:
    return (repo / '.git').exists()


def parse_iso8601(value: str):
    normalized = value.strip()
    if normalized.endswith('Z'):
        normalized = normalized[:-1] + '+00:00'
    return datetime.datetime.fromisoformat(normalized)


def load_tag_data(repo: Path):
    fmt = '%(refname:strip=2)%09%(creatordate:iso8601)'
    raw = run_git(repo, 'for-each-ref', '--sort=creatordate', f'--format={fmt}', 'refs/tags')
    rows = []
    for line in raw.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t', 1)
        if len(parts) != 2:
            continue
        name, date = parts[0], parts[1].strip()
        if not SEMVER_TAG_RE.match(name):
            continue
        dt = datetime.datetime.fromisoformat(date)
        rows.append((name, dt))
    rows.sort(key=lambda x: x[1])
    return rows


def parse_release_rows(raw_rows: str):
    rows = []
    for line in raw_rows.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t', 1)
        if len(parts) != 2:
            continue
        tag_name, published_at = parts[0].strip(), parts[1].strip()
        if not SEMVER_TAG_RE.match(tag_name):
            continue
        dt = parse_iso8601(published_at)
        rows.append((tag_name, dt))

    rows.sort(key=lambda x: x[1])
    deduped = []
    seen = set()
    for tag_name, dt in rows:
        if tag_name in seen:
            continue
        seen.add(tag_name)
        deduped.append((tag_name, dt))
    return deduped


def load_remote_release_data(repo_full_name: str):
    jq = '.[] | [.tag_name, (.published_at // .created_at)] | @tsv'
    raw = run_gh(
        'api',
        '--paginate',
        f'repos/{repo_full_name}/releases?per_page=100',
        '--jq',
        jq,
    )
    return parse_release_rows(raw)


def load_owner_repo_names(owner: str, limit: int, include_forks: bool, include_archived: bool):
    raw = run_gh(
        'repo',
        'list',
        owner,
        '--limit',
        str(limit),
        '--json',
        'nameWithOwner,isFork,isArchived',
    )
    rows = json.loads(raw)
    names = []
    for row in rows:
        if not include_forks and row.get('isFork'):
            continue
        if not include_archived and row.get('isArchived'):
            continue
        name = row.get('nameWithOwner')
        if name:
            names.append(name)
    return names


def count_feat_between_tags(repo: Path, tags):
    feat_counts = []
    for i in range(1, len(tags)):
        t_prev, _ = tags[i - 1]
        t_curr, d_curr = tags[i]
        log = run_git(repo, 'log', '--pretty=%s', f'{t_prev}..{t_curr}')
        feats = [line for line in log.split('\n') if FEAT_SUBJECT_RE.match(line)]
        feat_counts.append((t_curr, d_curr, len(feats)))
    return feat_counts


def cumulative_series(feat_counts):
    cum = []
    count = 0
    for _, d, n in feat_counts:
        count += n
        cum.append((d, count))
    return cum


def fit_gompertz(xs, ys):
    y_max = max(ys)
    best = None
    for K in [y_max * (1 + i / 80) for i in range(0, 241)]:
        if any(y >= K for y in ys):
            continue
        zs = []
        ok = True
        for t, y in zip(xs, ys):
            ratio = y / K
            if ratio <= 0 or ratio >= 1:
                ok = False
                break
            z = math.log(-math.log(ratio))
            zs.append(z)
        if not ok:
            continue
        n = len(xs)
        xbar = sum(xs) / n
        zbar = sum(zs) / n
        num = sum((x - xbar) * (z - zbar) for x, z in zip(xs, zs))
        den = sum((x - xbar) ** 2 for x in xs)
        if den == 0:
            continue
        b = -num / den
        ln_a = zbar + b * xbar
        a = math.exp(ln_a)
        sse = 0.0
        for t, y in zip(xs, ys):
            yhat = K * math.exp(-a * math.exp(-b * t))
            sse += (y - yhat) ** 2
        if best is None or sse < best[0]:
            best = (sse, K, a, b)
    return best


def fit_goel_okumoto(xs, ys):
    y_max = max(ys)
    best = None
    for a in [y_max * (1 + i / 80) for i in range(0, 241)]:
        if any(y >= a for y in ys):
            continue
        xs2 = []
        zs = []
        ok = True
        for t, y in zip(xs, ys):
            v = 1 - y / a
            if v <= 0:
                ok = False
                break
            xs2.append(t)
            zs.append(math.log(v))
        if not ok:
            continue
        den = sum(t * t for t in xs2)
        if den == 0:
            continue
        slope = sum(t * z for t, z in zip(xs2, zs)) / den
        b = -slope
        sse = 0.0
        for t, y in zip(xs, ys):
            yhat = a * (1 - math.exp(-b * t))
            sse += (y - yhat) ** 2
        if best is None or sse < best[0]:
            best = (sse, a, b)
    return best


def fit_weibull_nhpp(xs, ys):
    y_max = max(ys)
    best = None
    b_candidates = [50, 100, 200, 300, 500, 800, 1200, 1800, 2400, 3000, 3600]
    c_candidates = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    for a in [y_max * (1 + i / 80) for i in range(0, 241)]:
        if any(y >= a for y in ys):
            continue
        for b in b_candidates:
            for c in c_candidates:
                sse = 0.0
                for t, y in zip(xs, ys):
                    yhat = a * (1 - math.exp(-(t / b) ** c))
                    sse += (y - yhat) ** 2
                if best is None or sse < best[0]:
                    best = (sse, a, b, c)
    return best


def compute_fit_diagnostics(y_true, y_pred):
    sse = sum((y - yhat) ** 2 for y, yhat in zip(y_true, y_pred))
    rmse = math.sqrt(sse / len(y_true))
    mean_y = sum(y_true) / len(y_true)
    sst = sum((y - mean_y) ** 2 for y in y_true)
    r2 = 1.0 - (sse / sst) if sst > 0 else float('nan')
    return FitDiagnostics(sse=sse, rmse=rmse, r2=r2)


def render_png(
    plot_x,
    act,
    model_g,
    model_go,
    model_w,
    start_date,
    end_date,
    K,
    a_go,
    a_w,
    c_w,
    out_path: Path,
    title_prefix: str,
    actual_label: str,
):
    W, H = 1200, 720
    pad = 80
    img = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(img)

    x0, y0 = pad, H - pad
    x1, y1 = W - pad, pad
    draw.line((x0, y0, x1, y0), fill='black', width=2)
    draw.line((x0, y0, x0, y1), fill='black', width=2)

    y_max = max(max(act), max(model_g), max(model_go), max(model_w), 1.0)
    x_max = max(plot_x[-1], 1.0)

    def sx(t):
        return x0 + (t / x_max) * (x1 - x0)

    def sy(y):
        return y0 - (y / y_max) * (y0 - y1)

    for i in range(6):
        yv = y_max * i / 5
        y = sy(yv)
        draw.line((x0, y, x1, y), fill='#e6e6e6')
        draw.text((x0 - 10, y - 7), f'{int(yv)}', fill='black')

    def draw_series(series, color, width=2):
        pts = [(sx(t), sy(y)) for t, y in zip(plot_x, series)]
        if len(pts) > 2000:
            pts = pts[::2]
        draw.line(pts, fill=color, width=width)

    draw_series(act, 'black', 3)
    draw_series(model_g, '#1f77b4', 2)
    draw_series(model_go, '#ff7f0e', 2)
    draw_series(model_w, '#2ca02c', 2)

    draw.text((pad, 20), f'{title_prefix}: {start_date} to {end_date}', fill='black')

    legend = [
        (actual_label, 'black'),
        (f'Gompertz K={K:.1f}', '#1f77b4'),
        (f'NHPP GO a={a_go:.1f}', '#ff7f0e'),
        (f'NHPP Weibull a={a_w:.1f}, c={c_w:.2f}', '#2ca02c'),
    ]

    lx, ly = pad, H - pad + 10
    for i, (label, color) in enumerate(legend):
        y = ly + i * 22
        draw.rectangle((lx, y + 6, lx + 14, y + 14), fill=color)
        draw.text((lx + 22, y), label, fill='black')

    img.save(out_path)


def sanitize_source_name(source):
    source_text = str(source)
    if isinstance(source, Path):
        source_text = source.name if source.name else str(source)
    source_text = source_text.replace('/', '-')
    slug = re.sub(r'[^A-Za-z0-9._-]+', '-', source_text).strip('-')
    return slug if slug else 'repo'


def build_repo_slugs(repos):
    base_counts = {}
    for repo in repos:
        base = sanitize_source_name(repo)
        base_counts[base] = base_counts.get(base, 0) + 1

    slug_map = {}
    for repo in repos:
        base = sanitize_source_name(repo)
        if base_counts[base] == 1:
            slug_map[repo] = base
            continue
        digest = hashlib.sha1(str(repo).encode('utf-8')).hexdigest()[:8]
        slug_map[repo] = f'{base}-{digest}'
    return slug_map


def discover_git_repos(root: Path, recursive: bool):
    if not root.exists():
        return []

    found = []
    if is_git_repo(root):
        found.append(root.resolve())

    if recursive:
        for dot_git in root.rglob('.git'):
            repo = dot_git.parent
            if repo.is_dir():
                found.append(repo.resolve())
    else:
        for child in sorted(root.iterdir()):
            if child.is_dir() and is_git_repo(child):
                found.append(child.resolve())

    seen = set()
    deduped = []
    for repo in found:
        if repo not in seen:
            seen.add(repo)
            deduped.append(repo)
    return deduped


def collect_repos(repo_args, repos_root, recursive):
    repos = []
    for repo_arg in repo_args:
        repo = Path(repo_arg).expanduser().resolve()
        repos.append(repo)

    if repos_root:
        root = Path(repos_root).expanduser().resolve()
        repos.extend(discover_git_repos(root, recursive))

    if not repos:
        repos = [Path('.').resolve()]

    seen = set()
    deduped = []
    for repo in repos:
        if repo not in seen:
            seen.add(repo)
            deduped.append(repo)
    return deduped


def collect_remote_repos(remote_repos, owners, limit, include_forks, include_archived):
    repos = [normalize_remote_repo_name(repo) for repo in remote_repos if repo.strip()]

    for owner in owners:
        repos.extend(load_owner_repo_names(owner, limit, include_forks, include_archived))

    seen = set()
    deduped = []
    for repo in repos:
        if repo not in seen:
            seen.add(repo)
            deduped.append(repo)
    return deduped


def ensure_gh_available():
    if shutil.which(GH_BIN) is None and not (Path(GH_BIN).exists() and Path(GH_BIN).is_file()):
        raise SystemExit(f'gh CLI is required for remote mode (not found: {GH_BIN})')


def normalize_remote_repo_name(repo: str):
    normalized = repo.strip()
    if normalized.startswith('https://github.com/'):
        normalized = normalized[len('https://github.com/'):]
    if normalized.endswith('.git'):
        normalized = normalized[:-4]
    return normalized.strip('/')


def resolve_gh_bin(explicit_gh_bin: str | None):
    if explicit_gh_bin:
        return explicit_gh_bin
    if DEFAULT_HOMEBREW_GH.exists():
        return str(DEFAULT_HOMEBREW_GH)
    return 'gh'


def build_release_cumulative_series(releases):
    count = 0
    cum = []
    for _, published_at in releases:
        count += 1
        cum.append((published_at, count))
    return cum


def analyze_cumulative_series(
    source: str,
    out_path: Path,
    cum,
    point_count: int,
    observed_label: str,
    point_label: str,
    title_prefix: str,
    actual_label: str,
):
    if not cum:
        raise ValueError('Not enough data intervals')

    start = cum[0][0]
    xs = [(d - start).total_seconds() / 86400.0 for d, _ in cum]
    ys = [c for _, c in cum]

    filtered = [(x, y) for x, y in zip(xs, ys) if y > 0]
    xs = [x for x, _ in filtered]
    ys = [y for _, y in filtered]
    if len(xs) < 2:
        raise ValueError('Need at least two non-zero cumulative points')

    best_g = fit_gompertz(xs, ys)
    best_go = fit_goel_okumoto(xs, ys)
    best_w = fit_weibull_nhpp(xs, ys)
    if best_g is None or best_go is None or best_w is None:
        raise ValueError('Model fitting failed')

    K, a, b = best_g[1], best_g[2], best_g[3]
    a_go, b_go = best_go[1], best_go[2]
    a_w, b_w, c_w = best_w[1], best_w[2], best_w[3]

    def gompertz_y(t):
        return K * math.exp(-a * math.exp(-b * t))

    def go_y(t):
        return a_go * (1 - math.exp(-b_go * t))

    def weibull_y(t):
        return a_w * (1 - math.exp(-(t / b_w) ** c_w))

    model_g_obs = [gompertz_y(t) for t in xs]
    model_go_obs = [go_y(t) for t in xs]
    model_w_obs = [weibull_y(t) for t in xs]

    diagnostics = {
        'gompertz': compute_fit_diagnostics(ys, model_g_obs),
        'goel_okumoto': compute_fit_diagnostics(ys, model_go_obs),
        'weibull_nhpp': compute_fit_diagnostics(ys, model_w_obs),
    }

    max_t = max(1, int(math.ceil(xs[-1])))
    plot_x = list(range(0, max_t + 1))
    model_g = [gompertz_y(t) for t in plot_x]
    model_go = [go_y(t) for t in plot_x]
    model_w = [weibull_y(t) for t in plot_x]

    cum_ts = list(zip(xs, ys))

    def actual_at(t):
        y = 0
        for x, yv in cum_ts:
            if x <= t:
                y = yv
            else:
                break
        return y

    act = [actual_at(t) for t in plot_x]
    render_png(
        plot_x,
        act,
        model_g,
        model_go,
        model_w,
        start.date(),
        cum[-1][0].date(),
        K,
        a_go,
        a_w,
        c_w,
        out_path,
        title_prefix,
        actual_label,
    )
    return AnalysisResult(
        source=source,
        out_path=out_path,
        start_date=start.date(),
        end_date=cum[-1][0].date(),
        observed_count=ys[-1],
        observed_label=observed_label,
        point_count=point_count,
        point_label=point_label,
        gompertz_K=K,
        goel_okumoto_a=a_go,
        weibull_a=a_w,
        weibull_c=c_w,
        diagnostics=diagnostics,
    )


def analyze_repo(repo: Path, out_path: Path):
    if not is_git_repo(repo):
        raise ValueError('Not a git repository')

    tags = load_tag_data(repo)
    if len(tags) < 2:
        raise ValueError('Not enough semver tags')

    feat_counts = count_feat_between_tags(repo, tags)
    cum = cumulative_series(feat_counts)
    return analyze_cumulative_series(
        source=str(repo),
        out_path=out_path,
        cum=cum,
        point_count=len(tags),
        observed_label='Feature commits counted',
        point_label='Semver tags found',
        title_prefix='feature arrivals',
        actual_label='Actual (cumulative feat)',
    )


def analyze_remote_repo_release_rates(repo_full_name: str, out_path: Path):
    releases = load_remote_release_data(repo_full_name)
    if len(releases) < 2:
        raise ValueError('Not enough semver releases')

    cum = build_release_cumulative_series(releases)
    return analyze_cumulative_series(
        source=repo_full_name,
        out_path=out_path,
        cum=cum,
        point_count=len(releases),
        observed_label='Releases counted',
        point_label='Semver releases found',
        title_prefix='release arrivals',
        actual_label='Actual (cumulative releases)',
    )


def print_result(result: AnalysisResult):
    print(f'\n[{result.source}]')
    print(f'Date range: {result.start_date} -> {result.end_date}')
    print(f'{result.observed_label}: {result.observed_count}')
    print(f'{result.point_label}: {result.point_count}')
    print('Model asymptotes:')
    print(f'  Gompertz: K={result.gompertz_K:.3f}')
    print(f'  NHPP GO: a={result.goel_okumoto_a:.3f}')
    print(f'  NHPP Weibull: a={result.weibull_a:.3f}, c={result.weibull_c:.3f}')
    print('Fit diagnostics (on observed cumulative points):')
    for key, label in [
        ('gompertz', 'Gompertz'),
        ('goel_okumoto', 'NHPP GO'),
        ('weibull_nhpp', 'NHPP Weibull'),
    ]:
        diag = result.diagnostics[key]
        r2 = 'nan' if math.isnan(diag.r2) else f'{diag.r2:.4f}'
        print(f'  {label}: SSE={diag.sse:.3f}, RMSE={diag.rmse:.3f}, R2={r2}')
    print(f'PNG: {result.out_path}')


def main():
    global GH_BIN

    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', default=[], help='Path to a local repo (repeatable)')
    ap.add_argument('--repos-root', help='Directory containing local repos to analyze')
    ap.add_argument('--recursive', action='store_true', help='Recursively discover local repos under --repos-root')
    ap.add_argument('--remote', action='append', default=[], help='Remote repo `owner/name` for GH-only release-rate mode')
    ap.add_argument('--gh-owner', action='append', default=[], help='Analyze repos for owner/org via `gh repo list`')
    ap.add_argument('--gh-bin', help='Path to gh CLI binary (defaults to /opt/homebrew/bin/gh when available)')
    ap.add_argument('--gh-limit', type=int, default=200, help='Max repos fetched per --gh-owner')
    ap.add_argument('--gh-include-forks', action='store_true', help='Include fork repos in --gh-owner mode')
    ap.add_argument('--gh-include-archived', action='store_true', help='Include archived repos in --gh-owner mode')
    ap.add_argument('--out', default='feature-models.png', help='Output PNG path for single-repo mode')
    ap.add_argument('--out-dir', default='feature-models', help='Output directory for multi-repo mode')
    args = ap.parse_args()

    remote_mode = bool(args.remote or args.gh_owner)
    if remote_mode and (args.repo or args.repos_root):
        raise SystemExit('Use either local repo options (--repo/--repos-root) or GH options (--remote/--gh-owner), not both')

    targets = []
    analyze_fn = analyze_repo
    out_suffix = 'feature-models'

    if remote_mode:
        GH_BIN = resolve_gh_bin(args.gh_bin)
        ensure_gh_available()
        targets = collect_remote_repos(
            args.remote,
            args.gh_owner,
            args.gh_limit,
            args.gh_include_forks,
            args.gh_include_archived,
        )
        analyze_fn = analyze_remote_repo_release_rates
        out_suffix = 'release-models'
    else:
        targets = collect_repos(args.repo, args.repos_root, args.recursive)

    if not targets:
        raise SystemExit('No repositories to analyze')

    multi_repo = len(targets) > 1
    out_dir = Path(args.out_dir).resolve()
    if multi_repo:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    repo_slugs = build_repo_slugs(targets)
    single_out_name = args.out
    if remote_mode and not multi_repo and args.out == 'feature-models.png':
        single_out_name = 'release-models.png'

    for target in targets:
        out_path = Path(single_out_name).resolve()
        if multi_repo:
            out_path = out_dir / f'{repo_slugs[target]}-{out_suffix}.png'
        try:
            result = analyze_fn(target, out_path)
            results.append(result)
            print_result(result)
        except Exception as exc:
            failures.append((target, str(exc)))
            print(f'[SKIP] {target}: {exc}', file=sys.stderr)

    if not results:
        raise SystemExit('No repositories were successfully analyzed')

    if multi_repo:
        print(f'\nSuccessful analyses: {len(results)}')
    if failures:
        print(f'Skipped repos: {len(failures)}', file=sys.stderr)


if __name__ == '__main__':
    main()
