Example output for

```bash
smt store create-surrogates run_list/*
```

The hypothetical situation is as follows:

- The directory *run_list* contains a list of symbolic links to task files, numbered incrementally.
- The first 100 tasks were run successfully, but were not recorded in the Sumatra record store (the file was already open by another process).
  + ⇒ Their output exists, but the record store contains no record for them.
  + Exception: *run_list/task-55.json* points to a task description file which no longer exists.
- Tasks 100 and above were not run
  + ⇒ They have no associated output. Whether a corresponding record exists in the record store is irrelevant.

```
Surrogate records have been added for the following tasks:
  run_list/task-0.json
  run_list/task-10.json
  run_list/task-11.json
  run_list/task-12.json
  run_list/task-13.json
  run_list/task-14.json
  ...

The following task files will be removed:
  run_list/task-0.json     (∃ output, ∄ record)
  run_list/task-10.json    (∃ output, ∄ record)
  run_list/task-11.json    (∃ output, ∄ record)
  run_list/task-12.json    (∃ output, ∄ record)
  run_list/task-13.json    (∃ output, ∄ record)
  run_list/task-14.json    (∃ output, ∄ record)
  ...
  run_list/task-55.json    (∄ task desc)
  run_list/task-56.json    (∃ output, ∄ record)
  ...

The following task files will be kept since no corresponding output files were found:
  run_list/task-100.json
  run_list/task-101.json
  run_list/task-102.json
  run_list/task-103.json
  run_list/task-104.json
  run_list/task-105.json
  run_list/task-106.json
  ...
```
