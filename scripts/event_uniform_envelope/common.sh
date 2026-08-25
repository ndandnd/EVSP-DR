#!/bin/bash

evsp_die() {
  echo "ERROR: $*" >&2
  exit 2
}

evsp_require_unicorn() {
  [[ "$(id -un)" == "nc437" ]] || evsp_die "expected user nc437"
  [[ "$(hostname -s)" == unicorn-login-01* ]] \
    || evsp_die "expected unicorn-login-01"
}

evsp_repo_root() {
  git rev-parse --show-toplevel 2>/dev/null \
    || evsp_die "run this command from a Git checkout"
}

evsp_verify_remote_head() {
  local repo="$1"
  local branch="$2"
  local observed
  local count
  local remote_sha
  local remote_ref

  observed=$(git -C "$repo" ls-remote --heads origin "${branch}*") \
    || evsp_die "git ls-remote failed"
  printf '%s\n' "$observed" >&2
  count=$(printf '%s\n' "$observed" | awk 'NF {n++} END {print n+0}')
  [[ "$count" == "1" ]] \
    || evsp_die "expected exactly one remote branch matching $branch*"
  remote_sha=$(printf '%s\n' "$observed" | awk '{print $1}')
  remote_ref=$(printf '%s\n' "$observed" | awk '{print $2}')
  [[ "$remote_ref" == "refs/heads/$branch" ]] \
    || evsp_die "unexpected remote ref $remote_ref"
  [[ "$(git -C "$repo" rev-parse HEAD)" == "$remote_sha" ]] \
    || evsp_die "local HEAD is not the remote tip; git pull --ff-only first"
  printf '%s\n' "$remote_sha"
}

evsp_execution_checkout() {
  local repo="$1"
  local commit="$2"
  local checkout="$HOME/ladder-lite/execution/${commit}"
  mkdir -p "$HOME/ladder-lite/execution"
  if [[ ! -e "$checkout" ]]; then
    git -C "$repo" worktree add --detach "$checkout" "$commit" \
      >&2 || evsp_die "could not create execution checkout"
  fi
  [[ "$(git -C "$checkout" rev-parse HEAD)" == "$commit" ]] \
    || evsp_die "execution checkout commit mismatch"
  [[ -z "$(git -C "$checkout" status --porcelain)" ]] \
    || evsp_die "execution checkout is dirty"
  printf '%s\n' "$checkout"
}

evsp_submit_and_resolve() {
  local name="$1"
  shift
  local response
  local rc
  local attempt
  local ids

  set +e
  response=$(sbatch --parsable --job-name="$name" "$@" 2>&1)
  rc=$?
  set -e
  echo "sbatch[$name]: $response" >&2

  # ``--parsable`` is authoritative and survives jobs that start and finish
  # before they can ever be observed in squeue.  Retain the queue lookup only
  # as a compatibility fallback for nonstandard sbatch wrappers.
  if [[ "$rc" == 0 ]]; then
    local parsed_id="${response%%;*}"
    if [[ "$parsed_id" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$parsed_id"
      return 0
    fi
  fi

  for attempt in 1 2 3 4 5 6; do
    ids=$(
      squeue --me -h -o '%A|%j' |
        awk -F'|' -v name="$name" '$2 == name {print $1}' |
        sort -u
    )
    if [[ "$(printf '%s\n' "$ids" | awk 'NF {n++} END {print n+0}')" == "1" ]]; then
      printf '%s\n' "$ids"
      return 0
    fi
    sleep 2
  done
  evsp_die "could not resolve exactly one queued $name job; sbatch rc=$rc"
}
