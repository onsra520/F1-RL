# Portable evidence manifest paths

The shared backup/run-manifest path validator rejected POSIX traversal but accepted Windows drive-relative paths and NTFS stream syntax before constructing native Paths. Portable components now reject colon/Windows-invalid characters, control characters, trailing dot/space aliases, and reserved device basenames including superscript digit aliases. Both manifest creation and restore/model verification use the same boundary; valid nested filenames and the serialized manifest schema are unchanged.

A targeted regression went RED in nine cases before the fix. After adding reserved superscript aliases found by independent review,43 Windows manifest/run-manifest/trusted-signature tests passed. An initial Linux test invoked in a Windows Git worktree failed three provenance tests because its Git metadata contains Windows paths;38 other tests passed. This invocation is not Linux compatibility evidence. Exact-HEAD Linux verification must use a native clone.

No network backup, independent restore attestation, scientific authorization or trust bootstrap is introduced. Four permissions remain false.
