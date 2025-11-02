nitial Generation: You will perform a one-time analysis of the following source files: * wbs.md (Work Breakdown Structure) * uml.md (System diagrams and specifications) * report.md (Project scope and requirements)

Output File: Based on your analysis, you must generate a new file named tasklists.md.

Task Format: All items in tasklists.md must be formatted as Markdown checklists. * Incomplete: - [ ] Task description * Complete: - [x] Task description

State Management: You must maintain the state of this task list. When I (the user) notify you that a task is "complete," "finished," or "done," you will update tasklists.md by changing the corresponding task's prefix from - [ ] to - [x].

Persistence: This tasklists.md file will serve as our persistent record of progress.