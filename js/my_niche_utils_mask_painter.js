import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { MyNicheUtilsMessaging } from "./messaging.js";

// 実行状態管理
class MyNicheUtilsState {
    static isNodeRunning(nodeId) {
        return app.runningNodeId === nodeId;
    }

    static isAnyNodeRunning() {
        return !!app.runningNodeId;
    }    // ノードがユーザー入力待機状態かどうかを判定
    static isNodeWaitingForInput(node) {
        // 画像が表示されており、ボタンが存在する場合は入力待機状態の可能性がある
        return node && node.imgs && node.imgs.length > 0 &&
            node.continueButton && node.cancelButton &&
            node.isWaitingForUserInput; // 明示的な待機フラグを使用
    }
}

// 続行ボタンが押された時の処理
function continueButtonPressed(node) {
    console.log(`MyNicheUtils: Continue button pressed`);

    if (!node) {
        console.error("MyNicheUtils: Node reference is null");
        return;
    }

    const isRunning = MyNicheUtilsState.isNodeRunning(node.id);
    const isWaiting = MyNicheUtilsState.isNodeWaitingForInput(node);

    console.log(`MyNicheUtils: Node ID: ${node.id}, Button name: "${this.name}", Is running: ${isRunning}, Is waiting: ${isWaiting}`);

    // ボタンが有効で、実行中またはユーザー入力待機中の場合に動作
    if (this.name !== '' && (isRunning || isWaiting)) {
        console.log(`MyNicheUtils: Continue button pressed for node ${node.id}`);

        // MaskEditorの状態を確認
        console.log(`MyNicheUtils: Current images in node: ${node.imgs ? node.imgs.length : 0}`);
        if (node.imgs && node.imgs.length > 0) {
            node.imgs.forEach((img, index) => {
                console.log(`MyNicheUtils: Image ${index + 1} src: ${img.src}`);
                console.log(`MyNicheUtils: Image ${index + 1} complete: ${img.complete}`);
            });
        }

        MyNicheUtilsMessaging.sendContinue(node.id);

        // 待機状態を解除
        node.isWaitingForUserInput = false;

        // ボタンを無効化
        this.name = 'Processing...';
        if (node.cancelButton) {
            node.cancelButton.name = '';
        }

        node.setDirtyCanvas(true, true);
    } else {
        console.log(`MyNicheUtils: Continue button not active. Button name: "${this.name}", Running: ${isRunning}, Waiting: ${isWaiting}`);
    }
}

// キャンセルボタンが押された時の処理
function cancelButtonPressed(node) {
    console.log(`MyNicheUtils: Cancel button pressed`);

    if (!node) {
        console.error("MyNicheUtils: Node reference is null");
        return;
    }

    const isRunning = MyNicheUtilsState.isNodeRunning(node.id);
    const isWaiting = MyNicheUtilsState.isNodeWaitingForInput(node);

    console.log(`MyNicheUtils: Node ID: ${node.id}, Button name: "${this.name}", Is running: ${isRunning}, Is waiting: ${isWaiting}`);

    // ボタンが有効で、実行中またはユーザー入力待機中の場合に動作
    if (this.name !== '' && (isRunning || isWaiting)) {
        console.log(`MyNicheUtils: Cancel button pressed for node ${node.id}`);
        MyNicheUtilsMessaging.sendCancel(node.id);

        // 待機状態を解除
        node.isWaitingForUserInput = false;

        // ボタンを無効化
        this.name = 'Cancelling...';
        if (node.continueButton) {
            node.continueButton.name = '';
        }

        node.setDirtyCanvas(true, true);
    } else {
        console.log(`MyNicheUtils: Cancel button not active. Button name: "${this.name}", Running: ${isRunning}, Waiting: ${isWaiting}`);
    }
}

// ボタンのクリック効果を制御
function setupButtonBehavior(button) {
    // クリック効果の制御
    Object.defineProperty(button, 'clicked', {
        get: function () { return this._clicked; },
        set: function (v) { this._clicked = (v && this.name !== ''); }
    });

    // シリアライゼーションを無効化
    if (!button.options) button.options = {};
    button.options.serialize = false;
}

// プレビュー画像を表示する関数
function displayPreviewImages(event) {
    console.log(`MyNicheUtils: Received preview event for node ${event.detail.id}`, event.detail);

    const node = app.graph._nodes_by_id[event.detail.id];
    if (!node) {
        console.error(`MyNicheUtils: Node ${event.detail.id} not found`);
        console.log("MyNicheUtils: Available nodes:", Object.keys(app.graph._nodes_by_id));
        return;
    }

    console.log(`MyNicheUtils: Found node ${node.id}, comfyClass: ${node.comfyClass}`);

    if (node.comfyClass === "MyNicheUtilsMaskPainter") {
        console.log(`MyNicheUtils: Processing preview for MaskPainter node ${event.detail.id}`);

        // 現在の状態をログ出力
        console.log(`MyNicheUtils: Current button states - Continue: "${node.continueButton?.name || 'undefined'}", Cancel: "${node.cancelButton?.name || 'undefined'}", Waiting: ${node.isWaitingForUserInput}`);

        // 古い画像をクリア
        if (node.imgs) {
            console.log(`MyNicheUtils: Clearing ${node.imgs.length} old images`);
            node.imgs.forEach(img => {
                if (img.src) {
                    img.src = ''; // 古い画像のソースをクリア
                    img.onload = null; // イベントハンドラもクリア
                    img.onerror = null;
                }
            });
            node.imgs = [];
        }

        // 画像を表示
        showImages(node, event.detail.urls);

        // 以前のボタン状態をリセット（新しいプレビューの場合）
        // resetButtonStates(node); // 一時的にコメントアウト

        // ユーザー入力待機状態フラグを設定
        node.isWaitingForUserInput = true;

        // ボタンを有効化
        if (node.continueButton) {
            node.continueButton.name = "Continue with Applied Mask";
            console.log(`MyNicheUtils: Enabled continue button`);
        } else {
            console.error(`MyNicheUtils: Continue button not found on node`);
        }

        if (node.cancelButton) {
            node.cancelButton.name = "Cancel Operation";
            console.log(`MyNicheUtils: Enabled cancel button`);
        } else {
            console.error(`MyNicheUtils: Cancel button not found on node`);
        }

        // ノードを更新
        node.setDirtyCanvas(true, true);

        console.log(`MyNicheUtils: Preview setup complete for node ${event.detail.id}, waiting for user input`);
        console.log(`MyNicheUtils: Final button states - Continue: "${node.continueButton?.name}", Cancel: "${node.cancelButton?.name}", Waiting: ${node.isWaitingForUserInput}`);
    } else {
        console.log(`MyNicheUtils: Node ${event.detail.id} is not a MaskPainter (comfyClass: ${node.comfyClass})`);
    }
}

function showImages(node, urls) {
    if (!urls || urls.length === 0) {
        console.warn("MyNicheUtils: No images to display");
        return;
    }

    console.log(`MyNicheUtils: Loading ${urls.length} new images for node ${node.id}`);

    // 古い画像配列を完全にクリア
    if (node.imgs) {
        node.imgs.forEach(img => {
            if (img.onload) img.onload = null;
            if (img.onerror) img.onerror = null;
            if (img.src) img.src = '';
        });
    }
    // MaskEditorのためにnode.imgsを使用
    node.imgs = [];
    let loadedCount = 0;

    // カスタム描画フラグを設定（ComfyUIのデフォルト描画を制御するため）
    node.customImageDisplay = true;

    // キャンバスを一度クリアして重複描画を防ぐ
    node.setDirtyCanvas(true, true);

    urls.forEach((u, index) => {
        const img = new Image();
        node.imgs.push(img);

        // MaskEditorが必要とするメタデータを画像オブジェクトに追加
        img.setAttribute('data-filename', u.filename);
        img.setAttribute('data-subfolder', u.subfolder || '');
        img.setAttribute('data-type', u.type || 'input');

        img.onload = () => {
            loadedCount++;
            console.log(`MyNicheUtils: Loaded image ${index + 1}/${urls.length} (${u.filename})`);
            console.log(`MyNicheUtils: Image metadata - filename: ${img.getAttribute('data-filename')}, subfolder: ${img.getAttribute('data-subfolder')}, type: ${img.getAttribute('data-type')}`);

            if (loadedCount === urls.length) {
                console.log("MyNicheUtils: All images loaded, updating canvas");
                app.graph.setDirtyCanvas(true);

                // ノードサイズを調整
                adjustNodeSize(node);

                // 再描画を強制
                node.setDirtyCanvas(true, true);
            }
        };

        img.onerror = (error) => {
            console.error(`MyNicheUtils: Failed to load image ${index + 1} (${u.filename}):`, error);
        };

        // キャッシュバスターとタイムスタンプを追加
        const timestamp = Date.now();
        const cacheBuster = Math.random().toString(36).substring(2);
        img.src = api.apiURL(`/view?filename=${encodeURIComponent(u.filename)}&type=${u.type || 'input'}&subfolder=${u.subfolder || ''}&t=${timestamp}&cb=${cacheBuster}`);

        console.log(`MyNicheUtils: Loading image ${index + 1}: ${img.src}`);
    });
}

function adjustNodeSize(node) {
    if (!node.imgs || node.imgs.length === 0) return;

    const padding = 20;
    const maxImageHeight = 300;
    let totalHeight = 0;

    // 基本的なウィジェットの高さを計算
    const widgetHeight = node.widgets.length * 30 + 50;

    // 画像の高さを計算
    node.imgs.forEach(img => {
        if (img.complete) {
            const maxWidth = node.size[0] - padding * 2;
            const ratio = Math.min(maxWidth / img.width, maxImageHeight / img.height);
            totalHeight += img.height * ratio + padding;
        }
    });

    const newHeight = Math.max(200, widgetHeight + totalHeight + padding * 2);

    if (Math.abs(node.size[1] - newHeight) > 10) {
        node.size[1] = newHeight;
        console.log(`MyNicheUtils: Adjusted node size to ${node.size[0]}x${newHeight}`);
    }
}

// 背景描画の追加
function additionalDrawBackground(node, ctx) {
    // カスタム描画フラグが設定されている場合のみカスタム描画を実行
    if (!node.customImageDisplay || !node.imgs || node.imgs.length === 0) return;

    const padding = 10;
    const maxImageHeight = 300;
    let currentY = 100; // ウィジェットの下から開始

    // ウィジェットの最下部を計算
    if (node.widgets && node.widgets.length > 0) {
        const lastWidget = node.widgets[node.widgets.length - 1];
        currentY = lastWidget.last_y + 40;
    }

    // 画像描画領域全体をクリア（重複描画を防ぐ）
    const clearStartY = currentY - 10;
    const clearHeight = node.size[1] - clearStartY;
    ctx.clearRect(0, clearStartY, node.size[0], clearHeight);

    ctx.save();

    for (let i = 0; i < node.imgs.length; i++) {
        const img = node.imgs[i];
        if (img.complete && img.width > 0 && img.height > 0) {
            const maxWidth = node.size[0] - padding * 2;

            // アスペクト比を維持してリサイズ
            const ratio = Math.min(maxWidth / img.width, maxImageHeight / img.height);
            const imgWidth = img.width * ratio;
            const imgHeight = img.height * ratio;

            const imgX = (node.size[0] - imgWidth) / 2;
            const imgY = currentY;

            // 背景を描画
            ctx.fillStyle = "#2a2a2a";
            ctx.fillRect(imgX - 2, imgY - 2, imgWidth + 4, imgHeight + 4);

            // 画像を描画
            try {
                ctx.drawImage(img, imgX, imgY, imgWidth, imgHeight);
            } catch (error) {
                console.error("MyNicheUtils: Error drawing image:", error);
            }

            // 境界線を描画
            ctx.strokeStyle = "#4CAF50";
            ctx.lineWidth = 2;
            ctx.strokeRect(imgX, imgY, imgWidth, imgHeight);

            // 画像情報を表示
            ctx.fillStyle = "#ffffff";
            ctx.font = "12px Arial";
            ctx.fillText(`Mask Preview ${i + 1} (Click to edit)`, imgX, imgY - 5);

            currentY += imgHeight + padding;
        }
    }

    ctx.restore();
}

// ボタンの状態をリセットする関数
function resetButtonStates(node) {
    if (node.continueButton) {
        if (node.continueButton.name.includes('Processing') ||
            node.continueButton.name.includes('Cancelling') ||
            node.continueButton.name === 'Continue with Applied Mask') {
            node.continueButton.name = '';
            console.log(`MyNicheUtils: Reset continue button state for node ${node.id}`);
        }
    }

    if (node.cancelButton) {
        if (node.cancelButton.name.includes('Processing') ||
            node.cancelButton.name.includes('Cancelling') ||
            node.cancelButton.name === 'Cancel Operation') {
            node.cancelButton.name = '';
            console.log(`MyNicheUtils: Reset cancel button state for node ${node.id}`);
        }
    }

    // 待機状態もリセット
    node.isWaitingForUserInput = false;
}

// ComfyUI拡張機能として登録
app.registerExtension({
    name: "MyNicheUtils.MaskPainter",

    init() {
        // プレビューイベントリスナーを追加
        api.addEventListener("my-niche-utils-preview", displayPreviewImages);

        // ComfyUI Manager関連のコンソールエラーを抑制
        const originalConsoleError = console.error;
        console.error = function (...args) {
            const message = args.join(' ');
            if (message.includes('badge_mode') && message.includes('404')) {
                // badge_mode関連の404エラーは出力しない
                return;
            }
            originalConsoleError.apply(console, args);
        };

        // ネットワークリクエストを監視してマスクアップロードを検出
        const originalFetch = window.fetch;
        window.fetch = function (...args) {
            const url = args[0];
            if (typeof url === 'string' && url.includes('/upload/mask')) {
                console.log(`MyNicheUtils: Mask upload detected to:`, url);
                if (args[1] && args[1].body instanceof FormData) {
                    console.log(`MyNicheUtils: FormData detected in upload request`);
                }
            }
            return originalFetch.apply(this, args);
        };

        // 処理完了イベントリスナーを追加
        api.addEventListener("my-niche-utils-complete", (event) => {
            console.log(`MyNicheUtils: Received completion signal for node ${event.detail.id}`, event.detail);
            const node = app.graph._nodes_by_id[event.detail.id];
            if (node && node.comfyClass === "MyNicheUtilsMaskPainter") {
                console.log(`MyNicheUtils: Completion signal processed for node ${event.detail.id}, but NOT resetting to preserve state for next execution`);
                // リセットしない - 次回の実行でプレビュー表示時にリセットする
                // resetButtonStates(node);
                // node.setDirtyCanvas(true, true);
            }
        });

        // 実行開始時にボタン状態をリセットするためのイベントリスナー
        api.addEventListener("executing", (event) => {
            if (event.detail && app.graph) {
                const node = app.graph._nodes_by_id[event.detail];
                if (node && node.comfyClass === "MyNicheUtilsMaskPainter") {
                    console.log(`MyNicheUtils: Node ${event.detail} started executing, but NOT resetting button states to preserve functionality`);
                    // リセットしない - プレビュー表示時にのみリセットする
                    // resetButtonStates(node);
                    // node.setDirtyCanvas(true, true);
                }
            }
        });

        // 実行完了時にもボタン状態をリセット
        api.addEventListener("executed", (event) => {
            if (event.detail && event.detail.node && app.graph) {
                const node = app.graph._nodes_by_id[event.detail.node];
                if (node && node.comfyClass === "MyNicheUtilsMaskPainter") {
                    console.log(`MyNicheUtils: Node ${event.detail.node} execution completed, but NOT resetting button states to preserve functionality`);
                    // リセットしない - プレビュー表示時にのみリセットする
                    // resetButtonStates(node);
                    // node.setDirtyCanvas(true, true);
                }
            }
        });

        console.log("MyNicheUtils: MaskPainter extension initialized");
    },
    async nodeCreated(node) {
        if (node.comfyClass === "MyNicheUtilsMaskPainter") {
            console.log(`MyNicheUtils: MaskPainter node created: ${node.id}`);

            // プロパティを初期化
            node.imgs = [];
            node.isMyNicheUtilsNode = true;
            node.isWaitingForUserInput = false;
            node.customImageDisplay = false; // 初期状態ではComfyUIのデフォルト描画を使用

            // 続行ボタンを追加（ノード参照を直接保存）
            node.continueButton = node.addWidget("button", "", "", function () {
                continueButtonPressed.call(this, node);
            });
            setupButtonBehavior(node.continueButton);

            // キャンセルボタンを追加（ノード参照を直接保存）
            node.cancelButton = node.addWidget("button", "", "", function () {
                cancelButtonPressed.call(this, node);
            });
            setupButtonBehavior(node.cancelButton);

            console.log(`MyNicheUtils: Buttons created for node ${node.id}, continue button: ${!!node.continueButton}, cancel button: ${!!node.cancelButton}`);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MyNicheUtilsMaskPainter") {
            console.log("MyNicheUtils: Registering MaskPainter node definition");

            // 背景描画をオーバーライド
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function (ctx) {
                // カスタム描画フラグが設定されている場合、ComfyUIのデフォルト背景描画をスキップ
                if (this.customImageDisplay) {
                    // ウィジェットなどの基本的な描画のみ実行
                    if (onDrawBackground) {
                        // 一時的にimgsを空にしてデフォルト画像描画を防ぐ
                        const originalImgs = this.imgs;
                        this.imgs = [];
                        onDrawBackground.apply(this, arguments);
                        this.imgs = originalImgs;
                    }
                    // カスタム描画を実行
                    additionalDrawBackground(this, ctx);
                } else {
                    // 通常のComfyUI描画
                    if (onDrawBackground) {
                        onDrawBackground.apply(this, arguments);
                    }
                }
            };            // ノードの更新処理をオーバーライド
            const update = nodeType.prototype.update;
            nodeType.prototype.update = function () {
                if (update) {
                    update.apply(this, arguments);
                }

                // 実行状態に基づいてボタンの状態を更新
                if (this.continueButton && this.cancelButton) {
                    const isRunning = MyNicheUtilsState.isNodeRunning(this.id);
                    const isWaiting = MyNicheUtilsState.isNodeWaitingForInput(this);
                    const hasImages = this.imgs && this.imgs.length > 0;

                    // デバッグ情報（更新が多いので、状態変化時のみログ出力）
                    const currentContinueName = this.continueButton.name;
                    const currentCancelName = this.cancelButton.name;

                    // ユーザー入力待機中は常にボタンを有効にする
                    if (isWaiting && hasImages) {
                        if (this.continueButton.name === '' || this.continueButton.name.includes('Processing') || this.continueButton.name.includes('Cancelling')) {
                            this.continueButton.name = "Continue with Applied Mask";
                            console.log(`MyNicheUtils: Force-enabled continue button for node ${this.id} (waiting for input)`);
                        }
                        if (this.cancelButton.name === '' || this.cancelButton.name.includes('Processing') || this.cancelButton.name.includes('Cancelling')) {
                            this.cancelButton.name = "Cancel Operation";
                            console.log(`MyNicheUtils: Force-enabled cancel button for node ${this.id} (waiting for input)`);
                        }
                    }

                    // 状態変化をログ出力
                    if (currentContinueName !== this.continueButton.name || currentCancelName !== this.cancelButton.name) {
                        console.log(`MyNicheUtils: Button state changed for node ${this.id}: running=${isRunning}, waiting=${isWaiting}, hasImages=${hasImages}, continue="${this.continueButton.name}", cancel="${this.cancelButton.name}"`);
                    }
                }

                this.setDirtyCanvas(true, true);
            };
        }
    }
});
