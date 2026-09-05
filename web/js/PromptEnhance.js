import { app } from "../../../scripts/app.js";


const NONE_MODEL = "None";
const SELECT_WIDGET_NAME = "model_name_select";
const MIN_NODE_WIDTH = 420;
const MODELS_PROPERTY = "h3_prompt_models";
const SELECTED_PROPERTY = "h3_prompt_model_name";
const STORAGE_PREFIX = "h3-prompt-enhance:models:";


/** 将任意模型值规范成可用于下拉框的字符串。 */
function normalizeModelName(value) {
    const name = String(value ?? "").trim();
    return name && name.toLowerCase() !== NONE_MODEL.toLowerCase()
        ? name
        : NONE_MODEL;
}


/** 清洗后端返回的模型列表，保持顺序并去除重复项。 */
function normalizeModelNames(values) {
    const result = [];
    const seen = new Set();
    for (const value of Array.isArray(values) ? values : []) {
        const name = normalizeModelName(value);
        if (name === NONE_MODEL || seen.has(name)) continue;
        seen.add(name);
        result.push(name);
    }
    return result;
}


/** 标记工作流已变化，并刷新节点画布。 */
function markGraphDirty(node) {
    try {
        app.graph?.setDirtyCanvas?.(true, true);
        app.graph?.afterChange?.();
    } catch (error) {
        console.warn("H3PromptEnhance: 无法刷新工作流状态", error);
    }
    node.setDirtyCanvas?.(true, true);
}


/** 查找隐藏的后端 model_name 真值控件。 */
function findValueWidget(node) {
    return node.widgets?.find((widget) => widget?.name === "model_name") ?? null;
}


/** 查找模型服务基础地址控件。 */
function findBaseUrlWidget(node) {
    return node.widgets?.find((widget) => widget?.name === "base_url") ?? null;
}


/** 查找可见的动态模型下拉控件。 */
function findComboWidget(node) {
    return node.widgets?.find(
        (widget) => widget?.name === SELECT_WIDGET_NAME) ?? null;
}


/** 完全隐藏原生 STRING 控件及其 DOM，保留后端输入值。 */
function collapseWidget(valueWidget) {
    valueWidget.hidden = true;
    valueWidget.computeSize = () => [0, -4];

    const hideElement = () => {
        if (valueWidget.element) {
            valueWidget.element.style.display = "none";
        }
    };
    hideElement();
    requestAnimationFrame(hideElement);
    setTimeout(hideElement, 60);
}


/** 从旧工作流结构中读取原 model 字段。 */
function readLegacyModel(serializedNode) {
    const namedValue = serializedNode?.widgets_values_named?.model;
    const propertyValue = serializedNode?.properties?.model;
    return normalizeModelName(namedValue ?? propertyValue);
}


/** 规范基础地址，以便不同节点共享同一服务的本地模型缓存。 */
function normalizeBaseUrl(node) {
    return String(findBaseUrlWidget(node)?.value ?? "")
        .trim()
        .replace(/\/+$/, "");
}


/** 生成只依赖基础地址、不包含 API Key 的浏览器缓存键。 */
function buildStorageKey(node) {
    return STORAGE_PREFIX + encodeURIComponent(normalizeBaseUrl(node));
}


/** 从浏览器缓存安全读取模型列表。 */
function readLocalModels(node) {
    try {
        const raw = localStorage.getItem(buildStorageKey(node));
        const data = raw ? JSON.parse(raw) : null;
        return normalizeModelNames(data?.models);
    } catch (error) {
        console.warn("H3PromptEnhance: 无法读取本地模型缓存", error);
        return [];
    }
}


/** 将模型列表按基础地址安全写入浏览器缓存。 */
function writeLocalModels(node, models) {
    try {
        localStorage.setItem(buildStorageKey(node), JSON.stringify({ models }));
    } catch (error) {
        console.warn("H3PromptEnhance: 无法保存本地模型缓存", error);
    }
}


/** 读取工作流属性或浏览器缓存中的模型列表。 */
function readPersistedModels(node) {
    node.properties ||= {};
    const workflowModels = normalizeModelNames(
        node.properties[MODELS_PROPERTY]);
    return workflowModels.length ? workflowModels : readLocalModels(node);
}


/** 持久化模型列表和当前选择，但不保存 API Key。 */
function persistModelState(node, models, selected) {
    node.properties ||= {};
    const normalizedModels = normalizeModelNames(models);
    node.properties[MODELS_PROPERTY] = normalizedModels;
    node.properties[SELECTED_PROPERTY] = normalizeModelName(selected);
    writeLocalModels(node, normalizedModels);
}


/** 为尚未获取服务器列表的当前模型值创建临时选项。 */
function buildInitialOptions(current) {
    return current === NONE_MODEL
        ? [NONE_MODEL]
        : [NONE_MODEL, current];
}


/** 同步动态下拉选择到隐藏的后端真值控件。 */
function applySelection(node, selected, notify = true) {
    const valueWidget = findValueWidget(node);
    const comboWidget = findComboWidget(node);
    if (!valueWidget || !comboWidget) return;

    const normalized = normalizeModelName(selected);
    valueWidget.value = normalized;
    comboWidget.value = normalized;
    node.properties ||= {};
    node.properties[SELECTED_PROPERTY] = normalized;
    if (notify) markGraphDirty(node);
}


/** 将动态模型下拉移动到自动卸载开关之前。 */
function placeComboBeforeAutoUnload(node, comboWidget) {
    const autoUnloadWidget = node.widgets?.find(
        (widget) => widget?.name === "auto_unload");
    const comboIndex = node.widgets?.indexOf(comboWidget) ?? -1;
    const autoUnloadIndex = node.widgets?.indexOf(autoUnloadWidget) ?? -1;
    if (comboIndex < 0 || autoUnloadIndex < 0 || comboIndex < autoUnloadIndex) {
        return;
    }

    node.widgets.splice(comboIndex, 1);
    node.widgets.splice(autoUnloadIndex, 0, comboWidget);
}


/** 应用节点最小宽度，避免新节点因隐藏控件而过窄。 */
function applyMinimumNodeWidth(node) {
    const computedSize = node.computeSize?.();
    const width = Math.max(
        MIN_NODE_WIDTH,
        Number(computedSize?.[0]) || MIN_NODE_WIDTH,
        Number(node.size?.[0]) || MIN_NODE_WIDTH,
    );
    const height = Number(computedSize?.[1]) || Number(node.size?.[1]) || 0;
    node.setSize?.([width, height]);
}


/** 把已缓存模型列表恢复到动态下拉框。 */
function restoreModelOptions(node) {
    const valueWidget = findValueWidget(node);
    const comboWidget = findComboWidget(node);
    if (!valueWidget || !comboWidget) return;

    const models = readPersistedModels(node);
    const options = [NONE_MODEL, ...models];
    const propertySelection = normalizeModelName(
        node.properties?.[SELECTED_PROPERTY]);
    const widgetSelection = normalizeModelName(valueWidget.value);
    const preferred = propertySelection !== NONE_MODEL
        ? propertySelection
        : widgetSelection;
    const selected = options.includes(preferred) ? preferred : NONE_MODEL;

    comboWidget.options = comboWidget.options || {};
    comboWidget.options.values = options;
    valueWidget.value = selected;
    comboWidget.value = selected;
}


/** 监听基础地址变化，并恢复该服务对应的模型缓存。 */
function hookBaseUrlWidget(node) {
    const baseUrlWidget = findBaseUrlWidget(node);
    if (!baseUrlWidget || baseUrlWidget.h3PromptEnhanceHooked) return;

    baseUrlWidget.h3PromptEnhanceHooked = true;
    const originalCallback = baseUrlWidget.callback;
    baseUrlWidget.callback = function (value) {
        const result = originalCallback?.call(this, value);
        requestAnimationFrame(() => restoreModelOptions(node));
        return result;
    };
}


/** 创建真正的 LiteGraph combo 并隐藏原生 model_name 文本框。 */
function installModelCombo(node, serializedNode = null) {
    const valueWidget = findValueWidget(node);
    if (!valueWidget) return;

    node.properties ||= {};
    collapseWidget(valueWidget);

    const currentValue = normalizeModelName(valueWidget.value);
    const savedValue = normalizeModelName(
        node.properties[SELECTED_PROPERTY]);
    const legacyValue = readLegacyModel(serializedNode);
    const selected = currentValue !== NONE_MODEL
        ? currentValue
        : savedValue !== NONE_MODEL ? savedValue : legacyValue;

    let comboWidget = findComboWidget(node);
    if (!comboWidget) {
        comboWidget = node.addWidget("combo", "model_name_select", selected,
            (value) => applySelection(node, value), {
                values: [NONE_MODEL],
                serialize: false,
            });
        comboWidget.serialize = false;
        comboWidget.serializeValue = async () => undefined;
    }

    placeComboBeforeAutoUnload(node, comboWidget);
    comboWidget.options = comboWidget.options || {};
    comboWidget.options.values = buildInitialOptions(selected);
    applySelection(node, selected, false);
    restoreModelOptions(node);
    hookBaseUrlWidget(node);
    applyMinimumNodeWidth(node);
    node.setDirtyCanvas?.(true, true);
}


/** 使用执行结果中的模型列表更新、保存并显示下拉选项。 */
function updateModelOptions(node, values) {
    const valueWidget = findValueWidget(node);
    const comboWidget = findComboWidget(node);
    if (!valueWidget || !comboWidget) return;

    const models = normalizeModelNames(values);
    const options = [NONE_MODEL, ...models];
    const current = normalizeModelName(valueWidget.value);
    let selected = current;
    if (!options.includes(current)) selected = NONE_MODEL;

    comboWidget.options = comboWidget.options || {};
    comboWidget.options.values = options;
    valueWidget.value = selected;
    comboWidget.value = selected;
    persistModelState(node, models, selected);
    applyMinimumNodeWidth(node);
    markGraphDirty(node);
}


/** 从 ComfyUI 节点执行消息中提取模型数组。 */
function readExecutedModels(message) {
    const values = message?.models;
    if (Array.isArray(values?.[0])) return values[0];
    if (Array.isArray(values)) return values;
    return [];
}


/** 注册 H3PromptEnhance 节点的动态模型下拉生命周期。 */
app.registerExtension({
    name: "H3.PromptEnhance",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3PromptEnhance") return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installModelCombo(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (serializedNode) {
            const result = originalOnConfigure?.apply(this, arguments);
            installModelCombo(this, serializedNode);
            return result;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted?.apply(this, arguments);
            const models = readExecutedModels(message);
            if (models.length) updateModelOptions(this, models);
            return result;
        };
    },
});
