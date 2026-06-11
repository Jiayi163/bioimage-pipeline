"""CellProfiler-style dialog for adding modules to a pipeline."""

from __future__ import annotations

from collections.abc import Callable

from bioimage_pipeline.pipeline_catalog import (
    ModuleDefinition,
    get_module_definition,
    list_modules_by_category,
    search_modules,
)


def open_add_module_dialog(
    parent,
    *,
    on_add: Callable[[str], None],
    on_help: Callable[[ModuleDefinition], None] | None = None,
) -> None:
    """Open a modal-style window modeled on CellProfiler's AddModuleFrame."""
    import tkinter as tk
    from tkinter import ttk

    dialog = tk.Toplevel(parent)
    dialog.title("Add modules to pipeline")
    dialog.geometry("720x520")
    dialog.minsize(560, 400)
    dialog.transient(parent)
    dialog.grab_set()

    search_row = ttk.Frame(dialog, padding=8)
    search_row.pack(fill="x")
    ttk.Label(search_row, text="Find Modules:").pack(side="left")
    search_var = tk.StringVar()
    search_entry = ttk.Entry(search_row, textvariable=search_var)
    search_entry.pack(side="left", fill="x", expand=True, padx=(8, 8))

    body = ttk.Frame(dialog, padding=(8, 0, 8, 8))
    body.pack(fill="both", expand=True)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    ttk.Label(body, text="Module Categories", font=("Segoe UI", 9, "bold")).grid(
        row=0, column=0, sticky="nw",
    )
    ttk.Label(body, text="Modules", font=("Segoe UI", 9, "bold")).grid(
        row=0, column=1, sticky="nw",
    )
    category_column = ttk.Frame(body)
    category_column.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
    category_column.columnconfigure(0, weight=1)
    category_column.rowconfigure(0, weight=1)
    category_list = tk.Listbox(category_column, width=24, exportselection=False)
    category_list.grid(row=0, column=0, sticky="nsew")
    category_scroll = ttk.Scrollbar(category_column, orient="vertical", command=category_list.yview)
    category_scroll.grid(row=0, column=1, sticky="ns")
    category_list.configure(yscrollcommand=category_scroll.set)

    module_column = ttk.Frame(body)
    module_column.grid(row=1, column=1, sticky="nsew")
    module_column.columnconfigure(0, weight=1)
    module_column.rowconfigure(0, weight=1)
    module_list = tk.Listbox(module_column, exportselection=False)
    module_list.grid(row=0, column=0, sticky="nsew")
    module_scroll = ttk.Scrollbar(module_column, orient="vertical", command=module_list.yview)
    module_scroll.grid(row=0, column=1, sticky="ns")
    module_list.configure(yscrollcommand=module_scroll.set)

    grouped_modules = list(list_modules_by_category())
    modules_by_category = {category: mods for category, mods in grouped_modules}
    displayed_categories: list[str] = []

    def refresh_categories(filtered: list[tuple[str, list[ModuleDefinition]]] | None = None) -> None:
        category_list.delete(0, "end")
        displayed_categories.clear()
        source = filtered if filtered is not None else grouped_modules
        for category, mods in source:
            if mods:
                category_list.insert("end", f"{category} ({len(mods)})")
                displayed_categories.append(category)

    def selected_category_index() -> int | None:
        selection = category_list.curselection()
        return selection[0] if selection else None

    def refresh_module_list() -> None:
        module_list.delete(0, "end")
        index = selected_category_index()
        if index is None:
            return
        if index >= len(displayed_categories):
            return
        category = displayed_categories[index]
        query = search_var.get().strip()
        if query:
            allowed = {module.name for module in search_modules(query)}
            mods = [m for m in modules_by_category[category] if m.name in allowed]
        else:
            mods = modules_by_category[category]
        for module in mods:
            module_list.insert("end", module.name)
        if mods:
            module_list.selection_set(0)
            module_list.activate(0)

    def on_category_selected(_event: object | None = None) -> None:
        refresh_module_list()

    def selected_module_name() -> str | None:
        selection = module_list.curselection()
        if not selection:
            return None
        return module_list.get(selection[0])

    def add_selected() -> None:
        from tkinter import messagebox

        module_name = selected_module_name()
        if module_name is None and module_list.size() > 0:
            module_list.selection_set(0)
            module_name = module_list.get(0)
        if module_name is None:
            messagebox.showwarning(
                "Add module",
                "Select a module from the list on the right.",
                parent=dialog,
            )
            return
        dialog.grab_release()
        dialog.destroy()
        on_add(module_name)

    def show_help() -> None:
        module_name = selected_module_name()
        if module_name is None or on_help is None:
            return
        on_help(get_module_definition(module_name))

    def apply_search() -> None:
        query = search_var.get().strip()
        if not query:
            refresh_categories()
            category_list.selection_clear(0, "end")
            if displayed_categories:
                category_list.selection_set(0)
            refresh_module_list()
            return
        matches = search_modules(query)
        allowed = {module.name for module in matches}
        filtered = [
            (category, [m for m in mods if m.name in allowed])
            for category, mods in grouped_modules
        ]
        filtered = [(category, mods) for category, mods in filtered if mods]
        refresh_categories(filtered)
        category_list.selection_clear(0, "end")
        if filtered:
            category_list.selection_set(0)
        refresh_module_list()

    controls = ttk.Frame(dialog, padding=8)
    controls.pack(fill="x")
    ttk.Button(controls, text="+ Add to Pipeline", command=add_selected).pack(side="left")
    if on_help is not None:
        ttk.Button(controls, text="? Module Help", command=show_help).pack(
            side="left", padx=(8, 0),
        )
    ttk.Button(controls, text="Done", command=add_selected).pack(side="right")

    category_list.bind("<<ListboxSelect>>", on_category_selected)
    module_list.bind("<Double-1>", lambda _e: add_selected())
    search_entry.bind("<Return>", lambda _e: apply_search())
    ttk.Button(search_row, text="Search", command=apply_search).pack(side="left")

    refresh_categories()
    if displayed_categories:
        category_list.selection_set(0)
    refresh_module_list()
    search_entry.focus_set()
