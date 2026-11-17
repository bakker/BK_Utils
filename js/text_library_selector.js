import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Must match TEXT_LIBRARY_CATEGORIES in BK_Utils.py
const CATEGORY_DIRS = {
    "System Prompts": "system_prompts",
    "Wildcard Lists": "wildcard_lists",
};
const NO_FILES = "(no files found)";

// Combo specs come as [[values], opts] (legacy) or ["COMBO", {options: [...]}] (newer)
function valuesFromSpec(spec) {
    if (!spec) return [];
    if (Array.isArray(spec[0])) return spec[0];
    return spec[1]?.options ?? [];
}

// Ask the server for the current file list, so new files show up without a refresh
async function fetchAllFiles(fallback) {
    try {
        const res = await api.fetchApi("/object_info/TextLibrarySelector");
        const info = await res.json();
        return valuesFromSpec(info.TextLibrarySelector.input.required.file);
    } catch {
        return fallback;
    }
}

// Replace the contents of a combo's value list without replacing the array itself.
// A widget promoted onto a subgraph node can hold a reference to the inner widget's
// array; assigning a new array would leave that outer dropdown with a stale list.
function setComboValues(widget, values) {
    const current = widget.options?.values;
    if (Array.isArray(current)) {
        current.splice(0, current.length, ...values);
    } else if (widget.options) {
        widget.options.values = [...values];
    }
}

function applyFilter(node, allFiles, keepValue) {
    const categoryWidget = node.widgets?.find((w) => w.name === "category");
    const fileWidget = node.widgets?.find((w) => w.name === "file");
    if (!categoryWidget || !fileWidget) return;

    const dir = CATEGORY_DIRS[categoryWidget.value];
    const filtered = allFiles.filter((v) => v.replace(/\\/g, "/").startsWith(dir + "/"));
    const values = filtered.length ? filtered : [NO_FILES];

    setComboValues(fileWidget, values);
    // When loading a workflow, keep the saved value so a missing file gives a clear error
    if (!keepValue && !values.includes(fileWidget.value)) {
        fileWidget.value = values[0];
    }
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "BK_Utils.TextLibrarySelector",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TextLibrarySelector") return;
        const initialFiles = valuesFromSpec(nodeData.input?.required?.file);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const categoryWidget = node.widgets?.find((w) => w.name === "category");

            if (categoryWidget) {
                const originalCallback = categoryWidget.callback;
                categoryWidget.callback = function () {
                    const r = originalCallback?.apply(this, arguments);
                    fetchAllFiles(initialFiles).then((all) => applyFilter(node, all, false));
                    return r;
                };
            }
            // Keep the saved value here too: onNodeCreated also runs before a
            // workflow (or subgraph) restores its widget values
            applyFilter(node, initialFiles, true);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            fetchAllFiles(initialFiles).then((all) => applyFilter(this, all, true));
            return result;
        };
    },
});