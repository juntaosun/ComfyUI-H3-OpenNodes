import { app } from "../../../scripts/app.js";

/* H3MediaToVideo 单端口多连线：虚拟连线保存在节点属性，执行时注入隐藏输入。 */
const NODE_CLASS = "H3MediaToVideo";
const MEDIA_INPUT = "medias";
const BACKING_RE = /^media_[1-9]$/;
const MAX_MEDIA = 9;
const LINKS_PROPERTY = "h3_media_to_video_links";

/** 返回 H3MediaToVideo 的虚拟媒体连接记录。 */
function getLinks(node) {
    node.properties ||= {};
    if (!Array.isArray(node.properties[LINKS_PROPERTY])) node.properties[LINKS_PROPERTY] = [];
    return node.properties[LINKS_PROPERTY];
}

/** 获取图中的节点实例。 */
function getNode(graph, id) {
    return graph?.getNodeById?.(Number(id)) || app.graph?.getNodeById?.(Number(id));
}

/** 从原生连接对象读取来源节点和输出槽。 */
function readSource(graph, link) {
    if (!link) return null;
    const sourceId = link.origin_id ?? link.originId ?? link.from_id ?? link.fromId;
    const sourceNode = link.origin_node || link.originNode || link.fromNode || link.sourceNode || getNode(graph, sourceId);
    const sourceSlot = Number(link.origin_slot ?? link.originSlot ?? link.from_slot ?? link.fromSlot ?? 0);
    if (!sourceNode || !Number.isFinite(sourceSlot)) return null;
    return { sourceNode, sourceId: Number(sourceNode.id), sourceSlot, sourceType: link.type || sourceNode.outputs?.[sourceSlot]?.type || "H3_MEDIA" };
}

/** 返回可见 medias 输入及其索引。 */
function getMediaInput(node) {
    const input = node?.inputs?.find((item) => item?.name === MEDIA_INPUT);
    return input ? { input, index: node.inputs.indexOf(input) } : null;
}

/** 返回唯一可见 medias 端口的画布坐标。 */
function getMediaPosition(node) {
    const media = getMediaInput(node);
    if (!media) return null;
    const point = node.getInputPos?.(media.index);
    if (Array.isArray(point)) return point;
    const result = [0, 0];
    try {
        const legacy = node.getConnectionPos?.(true, media.index, result);
        return Array.isArray(legacy) ? legacy : result;
    } catch (error) {
        return [Number(node.pos?.[0] || 0), Number(node.pos?.[1] || 0) + 40 + media.index * 20];
    }
}

/** 清理重复或不存在来源节点的虚拟连接。 */
function normalizeLinks(node) {
    const graph = node?.graph || app.graph;
    const seen = new Set();
    const valid = getLinks(node).filter((link) => {
        const sourceId = Number(link?.source_id);
        const sourceSlot = Number(link?.source_slot);
        const key = `${sourceId}:${sourceSlot}`;
        if (!Number.isFinite(sourceId) || !Number.isFinite(sourceSlot) || seen.has(key)) return false;
        if (graph?.getNodeById && !getNode(graph, sourceId)) return false;
        seen.add(key);
        return true;
    });
    valid.forEach((link, index) => { link.order = index + 1; });
    node.properties[LINKS_PROPERTY] = valid.slice(0, MAX_MEDIA);
    return node.properties[LINKS_PROPERTY];
}

/** 删除前端定义中的隐藏传输输入，避免它们撑高节点。 */
function trimNodeDefinition(nodeData) {
    const remove = (container) => {
        if (!container || typeof container !== "object") return;
        for (const name of Object.keys(container)) {
            if (BACKING_RE.test(name)) delete container[name];
        }
    };
    remove(nodeData?.input?.required);
    remove(nodeData?.input?.optional);
    remove(nodeData?.required);
    remove(nodeData?.optional);
    for (const key of ["required", "optional"]) {
        if (Array.isArray(nodeData?.input_order?.[key])) {
            nodeData.input_order[key] = nodeData.input_order[key].filter((name) => !BACKING_RE.test(String(name)));
        }
    }
    if (Array.isArray(nodeData?.inputs)) {
        nodeData.inputs = nodeData.inputs.filter((input) => !BACKING_RE.test(String(input?.name || input?.id || input || "")));
    }
}

/** 将新原生连接记录为虚拟连接，并断开可见端口上的临时连接。 */
function convertVisibleConnection(node, linkInfo) {
    const media = getMediaInput(node);
    if (!media?.input?.link && !linkInfo) return false;
    const graph = node.graph || app.graph;
    const nativeLink = linkInfo || graph?.links?.get?.(media.input.link) || graph?._links?.[media.input.link];
    const source = readSource(graph, nativeLink);
    if (!source || source.sourceNode === node) return false;
    const links = normalizeLinks(node);
    if (links.some((item) => Number(item.source_id) === source.sourceId && Number(item.source_slot) === source.sourceSlot)) {
        node.disconnectInput?.(media.index);
        return false;
    }
    if (links.length >= MAX_MEDIA) {
        node.disconnectInput?.(media.index);
        return false;
    }
    links.push({ source_id: source.sourceId, source_slot: source.sourceSlot, source_type: source.sourceType, order: links.length + 1 });
    node.properties[LINKS_PROPERTY] = links;
    node.disconnectInput?.(media.index);
    node.setDirtyCanvas?.(true, true);
    graph?.setDirtyCanvas?.(true, true);
    graph?.change?.();
    return true;
}

/** 获取来源节点输出端口位置。 */
function getOutputPosition(node, slot) {
    const point = node?.getOutputPos?.(slot);
    if (Array.isArray(point)) return point;
    const result = [0, 0];
    try {
        const legacy = node?.getConnectionPos?.(false, slot, result);
        return Array.isArray(legacy) ? legacy : result;
    } catch (error) {
        return [Number(node?.pos?.[0] || 0) + Number(node?.size?.[0] || 160), Number(node?.pos?.[1] || 0) + 40 + slot * 20];
    }
}

/** 绘制虚拟媒体连线和每条线的连续序号。 */
function drawVirtualLinks(canvas, ctx) {
    const graph = canvas?.graph || app.graph;
    if (!ctx || !graph?._nodes || canvas.links_render_mode === globalThis.LiteGraph?.HIDDEN_LINK) return;
    for (const target of graph._nodes) {
        if (target?.comfyClass !== NODE_CLASS && target?.type !== NODE_CLASS) continue;
        const targetPoint = getMediaPosition(target);
        if (!targetPoint) continue;
        for (const [index, item] of normalizeLinks(target).entries()) {
            const sourceNode = getNode(graph, item.source_id);
            const sourcePoint = getOutputPosition(sourceNode, Number(item.source_slot));
            if (!sourceNode || !sourcePoint) continue;
            const midX = (sourcePoint[0] + targetPoint[0]) / 2;
            const midY = (sourcePoint[1] + targetPoint[1]) / 2;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(sourcePoint[0], sourcePoint[1]);
            ctx.bezierCurveTo(sourcePoint[0] + 80, sourcePoint[1], targetPoint[0] - 80, targetPoint[1], targetPoint[0], targetPoint[1]);
            ctx.lineWidth = canvas.connections_width || 3;
            ctx.strokeStyle = globalThis.LGraphCanvas?.link_type_colors?.H3_MEDIA || "#34d399";
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(midX, midY, 8, 0, Math.PI * 2);
            ctx.fillStyle = "#34d399";
            ctx.fill();
            ctx.fillStyle = "#071510";
            ctx.font = "bold 10px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(String(index + 1), midX, midY);
            ctx.restore();
        }
    }
}

/** 将浏览器鼠标事件转换为画布坐标。 */
function getGraphPosition(canvas, event) {
    try {
        canvas.adjustMouseEvent?.(event);
    } catch (error) {
        // 兼容不提供 adjustMouseEvent 的旧版 LiteGraph。
    }
    if (Array.isArray(canvas?.graph_mouse)) return [canvas.graph_mouse[0], canvas.graph_mouse[1]];
    if (Number.isFinite(event?.canvasX) && Number.isFinite(event?.canvasY)) return [event.canvasX, event.canvasY];
    const rect = canvas?.canvas?.getBoundingClientRect?.();
    const scale = canvas?.ds?.scale || 1;
    const offset = canvas?.ds?.offset || [0, 0];
    if (rect && Number.isFinite(event?.clientX) && Number.isFinite(event?.clientY)) {
        return [(event.clientX - rect.left) / scale - offset[0], (event.clientY - rect.top) / scale - offset[1]];
    }
    return [0, 0];
}

/** 将画布坐标转换为浏览器客户区坐标。 */
function getClientPosition(canvas, point) {
    const rect = canvas?.canvas?.getBoundingClientRect?.();
    if (!rect) return null;
    const scale = canvas?.ds?.scale || 1;
    const offset = canvas?.ds?.offset || [0, 0];
    return { x: rect.left + (point[0] + offset[0]) * scale, y: rect.top + (point[1] + offset[1]) * scale };
}

/** 查找鼠标位置最近的虚拟媒体连线。 */
function hitTestVirtualLinks(graph, x, y) {
    let best = null;
    for (const targetNode of graph?._nodes || []) {
        if (targetNode?.comfyClass !== NODE_CLASS && targetNode?.type !== NODE_CLASS) continue;
        const targetPoint = getMediaPosition(targetNode);
        if (!targetPoint) continue;
        normalizeLinks(targetNode).forEach((link, index) => {
            const sourceNode = getNode(graph, link.source_id);
            const sourcePoint = getOutputPosition(sourceNode, Number(link.source_slot));
            if (!sourceNode || !sourcePoint) return;
            const mid = [(sourcePoint[0] + targetPoint[0]) / 2, (sourcePoint[1] + targetPoint[1]) / 2];
            const distance = Math.hypot(x - mid[0], y - mid[1]);
            if (distance <= 18 && (!best || distance < best.distance)) {
                best = { targetNode, index, point: mid, distance };
            }
        });
    }
    return best;
}

/** 删除指定的虚拟媒体连线并通知画布和工作流更新。 */
function removeVirtualLink(targetNode, index) {
    const links = normalizeLinks(targetNode);
    if (index < 0 || index >= links.length) return false;
    links.splice(index, 1);
    normalizeLinks(targetNode);
    targetNode.setDirtyCanvas?.(true, true);
    (targetNode.graph || app.graph)?.setDirtyCanvas?.(true, true);
    (targetNode.graph || app.graph)?.change?.();
    return true;
}

/** 在虚拟连线位置打开 ComfyUI 原生删除菜单。 */
function openLinkMenu(canvas, hit, event) {
    const anchor = getClientPosition(canvas, hit.point) || { x: event?.clientX || 0, y: event?.clientY || 0 };
    const menuEvent = typeof PointerEvent === "function"
        ? new PointerEvent("pointerdown", { clientX: anchor.x + 8, clientY: anchor.y + 8, bubbles: true, cancelable: true })
        : new MouseEvent("mousedown", { clientX: anchor.x + 8, clientY: anchor.y + 8, bubbles: true, cancelable: true });
    let menuInstance = null;
    const remove = () => {
        removeVirtualLink(hit.targetNode, hit.index);
        menuInstance?.close?.();
        menuInstance?.remove?.();
    };
    if (globalThis.LiteGraph?.ContextMenu) {
        menuInstance = new globalThis.LiteGraph.ContextMenu([
            { content: "删除连线", callback: remove },
        ], { event: menuEvent });
    }
}

/** 判断当前画布是否正在创建新的媒体连线。 */
function isConnectingMedia(canvas) {
    const node = canvas?.connecting_node || canvas?.connectingNode;
    const input = canvas?.connecting_input || canvas?.connectingInput;
    if (!node || !input) return false;
    const slot = typeof input === "number" ? node.inputs?.[input] : input;
    return String(slot?.name || "") === MEDIA_INPUT;
}

/** 覆盖画布连接层，补绘虚拟连接并恢复其菜单交互。 */
function patchCanvas() {
    const canvas = app.canvas;
    if (!canvas || canvas.__h3MediaToVideoPatched || typeof canvas.drawConnections !== "function") return;
    canvas.__h3MediaToVideoPatched = true;
    const originalDraw = canvas.drawConnections;
    canvas.drawConnections = function (ctx) {
        const result = originalDraw.apply(this, arguments);
        drawVirtualLinks(this, ctx || this.bgctx || this.ctx);
        return result;
    };
    const originalDown = canvas.processMouseDown;
    canvas.processMouseDown = function (event) {
        if (!isConnectingMedia(this)) {
            const [x, y] = getGraphPosition(this, event);
            const hit = hitTestVirtualLinks(this.graph || app.graph, x, y);
            if (hit) {
                openLinkMenu(this, hit, event);
                event?.preventDefault?.();
                event?.stopImmediatePropagation?.();
                return true;
            }
        }
        return originalDown?.apply(this, arguments);
    };
    const linkPointerHandler = (event) => {
        if (isConnectingMedia(canvas)) return;
        const [x, y] = getGraphPosition(canvas, event);
        const hit = hitTestVirtualLinks(canvas.graph || app.graph, x, y);
        if (!hit) return;
        openLinkMenu(canvas, hit, event);
        event.preventDefault?.();
        event.stopPropagation?.();
        event.stopImmediatePropagation?.();
    };
    canvas.canvas?.addEventListener?.("pointerdown", linkPointerHandler, true);
}

/** 在 graphToPrompt 阶段把虚拟连接转换为后端隐藏输入。 */
function patchGraphToPrompt() {
    if (app.__h3MediaToVideoPromptPatched || typeof app.graphToPrompt !== "function") return;
    app.__h3MediaToVideoPromptPatched = true;
    const original = app.graphToPrompt;
    app.graphToPrompt = async function () {
        const output = await original.apply(this, arguments);
        const prompt = output?.output || output || {};
        for (const node of app.graph?._nodes || []) {
            if (node?.comfyClass !== NODE_CLASS && node?.type !== NODE_CLASS) continue;
            const promptNode = prompt[String(node.id)];
            if (!promptNode) continue;
            promptNode.inputs ||= {};
            for (let index = 1; index <= MAX_MEDIA; index += 1) delete promptNode.inputs[`media_${index}`];
            let mediaIndex = 1;
            for (const link of normalizeLinks(node)) {
                // 被忽略的上游节点不会出现在最终 prompt 中，不能继续作为后端连接提交。
                if (!Object.prototype.hasOwnProperty.call(prompt, String(link.source_id))) continue;
                promptNode.inputs[`media_${mediaIndex}`] = [String(link.source_id), Number(link.source_slot)];
                mediaIndex += 1;
            }
            delete promptNode.inputs.medias;
        }
        return output;
    };
}

/** 安装节点生命周期钩子，保证单端口连接可重复接入。 */
function installNode(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS || nodeType.prototype.__h3MediaToVideoInstalled) return;
    trimNodeDefinition(nodeData);
    nodeType.prototype.__h3MediaToVideoInstalled = true;
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        normalizeLinks(this);
        return result;
    };
    const originalConfigured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = originalConfigured?.apply(this, arguments);
        normalizeLinks(this);
        this.setDirtyCanvas?.(true, true);
        return result;
    };
    const originalConnections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
        const result = originalConnections?.apply(this, arguments);
        const input = this.inputs?.[Number(index)];
        if (type === 1 && connected && input?.name === MEDIA_INPUT) {
            setTimeout(() => convertVisibleConnection(this, linkInfo), 0);
        }
        return result;
    };
    const originalDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
        const result = originalDraw?.apply(this, arguments);
        normalizeLinks(this);
        return result;
    };
}

/** 注册扩展并在 ComfyUI 画布初始化后安装补丁。 */
app.registerExtension({
    name: "H3.MediaToVideo",
    setup() {
        const installPatches = (attempt = 0) => {
            patchGraphToPrompt();
            patchCanvas();
            if (attempt < 8 && (!app.__h3MediaToVideoPromptPatched || !app.canvas?.__h3MediaToVideoPatched)) {
                setTimeout(() => installPatches(attempt + 1), 250);
            }
        };
        installPatches();
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        installNode(nodeType, nodeData);
    },
});
