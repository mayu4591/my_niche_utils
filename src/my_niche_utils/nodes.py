from inspect import cleandoc
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import os
import json
import torch
import hashlib
import sys
import logging
import zipfile
import glob
from collections import Counter

# いるか不明。とりあえず入れておく
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))))

import folder_paths
import node_helpers

from nodes import ImageBatch
from server import PromptServer
import time

# メッセージホルダークラス（cg-image-pickerから参考）
class MessageHolder:
    messages = {}
    cancelled = False

    @classmethod
    def addMessage(cls, id, message):
        cls.messages[str(id)] = message

    @classmethod
    def waitForMessage(cls, id, period=0.1):
        sid = str(id)
        while not (sid in cls.messages) and not ("-1" in cls.messages):
            if cls.cancelled:
                cls.cancelled = False
                raise Exception("Cancelled")
            time.sleep(period)
        if cls.cancelled:
            cls.cancelled = False
            raise Exception("Cancelled")
        message = cls.messages.pop(str(id), None) or cls.messages.pop("-1")
        return message

# サーバールートの追加
try:
    from aiohttp import web
    routes = PromptServer.instance.routes

    @routes.post('/my_niche_utils_message')
    async def make_selection(request):
        post = await request.post()
        MessageHolder.addMessage(post.get("id"), post.get("message"))
        return web.json_response({})

    logging.info("MyNicheUtils: Server routes registered successfully")
except ImportError:
    logging.error("MyNicheUtils: Failed to import aiohttp")
except Exception as e:
    logging.error(f"MyNicheUtils: Failed to register server routes: {e}")


class MyNicheUtilsSaveImage:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        self.compress_level = 4
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "images": ("IMAGE", { "tooltip": "This is an image"}),
                "output_dir": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                }),
                "file_name": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                }),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "MyNicheUtils"

    def save_images(self, images, output_dir, file_name, prompt=None, extra_pnginfo=None):
        # imageをoutput_dir/file_nameに保存する
        images = self._save_images(images, file_name, output_dir, prompt, extra_pnginfo)

        return (images,)

    # nodes.pyの修正版
    def _save_images(self, images, filename=None, output_dir=None, prompt=None, extra_pnginfo=None):
#        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
#            if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # filenameの拡張子は事前に削除
            filename, ext = os.path.splitext(filename)
            # extがNoneでない場合はログ出力
            if not ext is None:
               logging.info(f"{filename}.{ext}: delete extention")

            file = f"{filename}.png"
            img.save(os.path.join(output_dir, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
#                "subfolder": subfolder,
                "type": self.type
            })
#            counter += 1

        return { "ui": { "images": results } }

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# nodes.pyの修正版
class MyNicheUtilsLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),},
                 "optional":{
                    "target_dir": ("STRING", {
                        "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    }),}
                }

    CATEGORY = "MyNicheUtils"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "IMAGE_PATH", "FILENAME")
    FUNCTION = "load_image"
    def load_image(self, image, target_dir=None):
        image_path = folder_paths.get_annotated_filepath(image)
        if target_dir is not None and target_dir != "":
            image_path = os.path.join(target_dir, os.path.basename(image_path))

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # 拡張子無しのファイル名を取得
        filename = os.path.splitext(os.path.basename(image_path))[0]

        return (output_image, output_mask, image_path, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class MyNicheUtilsImageBatch(ImageBatch):

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image1": ("IMAGE",), "image2": ("IMAGE",)},
            "optional": { "image3": ("IMAGE",), "image4": ("IMAGE",), "image5": ("IMAGE",), "image6": ("IMAGE",),
                        "image7": ("IMAGE",), "image8": ("IMAGE",), "image9": ("IMAGE",), "image10": ("IMAGE",),
                        "image11": ("IMAGE",), "image12": ("IMAGE",), "image13": ("IMAGE",), "image14": ("IMAGE",),
                        "image15": ("IMAGE",), "image16": ("IMAGE",), }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"

    CATEGORY = "MyNicheUtils"

    def batch(self, image1, image2, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, image11=None, image12=None, image13=None, image14=None, image15=None, image16=None):
        # image3-16をリスト化、Noneを除去
        images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, image13, image14, image15, image16]
        images = [img for img in images if img is not None]

        # imagesをループしてsuper.batchを呼び出す
        for i, img in enumerate(images):
            if i == 0:
                s = img
            else:
                s = super().batch(s, img)[0]

        return (s,)

# was-node-suite-comfyui, Nuber Counterの修正版
class MyNicheUtilsCounter:
    def __init__(self):
        self.counters = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_type": (["integer", "float"],),
                "mode": (["increment", "decrement", "increment_to_end", "decrement_to_end","increment_loop", "decrement_loop"],),
                "start": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01}),
                "end": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01}),
                "step": ("FLOAT", {"default": 1, "min": 0, "max": 99999, "step": 0.01}),
            },
            "optional": {
                "reset_bool": ("NUMBER",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    RETURN_TYPES = ("NUMBER", "FLOAT", "INT")
    RETURN_NAMES = ("number", "float", "int")
    FUNCTION = "increment_number"

    CATEGORY = "MyNicheUtils"

    def increment_number(self, number_type, mode, start, end, step, unique_id, reset_bool=0):

        counter = int(start) if mode == 'integer' else start
        if self.counters.__contains__(unique_id):
            counter = self.counters[unique_id]

        if round(reset_bool) >= 1:
            counter = start

        if mode == 'increment':
            counter += step
        elif mode == 'deccrement':
            counter -= step
        elif mode == 'increment_to_end':
            counter = counter + step if counter < end else counter
        elif mode == 'decrement_to_end':
            counter = counter - step if counter > end else counter
        elif mode == 'increment_to_loop':
            counter = (counter + step) % end
        elif mode == 'decrement_to_loop':
            counter = (counter - step) % end

        self.counters[unique_id] = counter

        result = int(counter) if number_type == 'integer' else float(counter)

        return ( result, float(counter), int(counter) )

class MyNicheUtilsMaskPainter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"image": ("IMAGE",),
                     "mask": ("MASK",),},
                 "optional":{
                    "enable_preview": ("BOOLEAN", {"default": True}),
                    "auto_continue": ("BOOLEAN", {"default": False}),},
                 "hidden": {"unique_id": "UNIQUE_ID"},
                }

    CATEGORY = "MyNicheUtils"

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "mask_painter"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 常に実行されるように、毎回異なる値を返す
        import time
        return str(time.time())

    def mask_painter(self, image, mask, unique_id, enable_preview=True, auto_continue=False):
        logging.info(f"MyNicheUtils: MaskPainter execution started for node {unique_id}, enable_preview={enable_preview}, auto_continue={auto_continue}")

        # 入力されたImageとMaskを処理
        batch_size = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]

        logging.info(f"MyNicheUtils: Processing image batch_size={batch_size}, height={height}, width={width}")

        # Maskのサイズを画像に合わせて調整
        if mask.shape[1] != height or mask.shape[2] != width:
            # マスクをリサイズ
            mask_resized = torch.nn.functional.interpolate(
                mask.unsqueeze(1),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        else:
            mask_resized = mask

        # マスクをバッチサイズに合わせる
        if mask_resized.shape[0] != batch_size:
            if mask_resized.shape[0] == 1:
                mask_resized = mask_resized.repeat(batch_size, 1, 1)
            else:
                mask_resized = mask_resized[:batch_size]

        # プレビュー用の画像を作成（入力画像にマスクをアルファチャンネルとして追加）
        # 常にプレビューを表示する（enable_previewパラメータは将来の拡張用）
        if True:  # enable_preview:
            logging.info(f"MyNicheUtils: Creating preview images for node {unique_id}")
            preview_images = []
            for i in range(batch_size):
                # RGB画像を取得
                rgb_image = image[i].cpu().numpy()
                # マスクを取得（0-1の範囲）
                alpha_mask = mask_resized[i].cpu().numpy()

                # RGBA形式に変換（元画像にマスクをアルファチャンネルとして設定）
                rgba_image = np.zeros((height, width, 4), dtype=np.float32)

                # 元の画像をそのまま設定
                rgba_image[:, :, :3] = rgb_image  # RGB（元画像のまま）

                # マスクをアルファチャンネルに設定
                # MaskEditorで編集できるように、マスクをアルファチャンネルとして設定
                # アルファチャンネル: 1.0 = 不透明（マスク対象）、0.0 = 透明（マスク対象外）
                # ComfyUIマスク: 1.0 = マスク対象外、0.0 = マスク対象 → 反転が必要
                rgba_image[:, :, 3] = 1.0 - alpha_mask  # Alpha（マスクを反転してアルファチャンネルに）

                # PIL Imageに変換してプレビュー用に保存
                preview_img = Image.fromarray((rgba_image * 255).astype(np.uint8), 'RGBA')

                # 一時的なプレビューファイルとして保存
                input_dir = folder_paths.get_input_directory()
                preview_filename = f"mask_preview_{unique_id}_{i}.png"
                preview_path = os.path.join(input_dir, preview_filename)

                # アルファチャンネルを確実に保存するため、PNGInfoを使用
                metadata = PngInfo()
                metadata.add_text("source", "MyNicheUtilsMaskPainter")
                metadata.add_text("unique_id", str(unique_id))
                metadata.add_text("batch_index", str(i))

                preview_img.save(preview_path, format='PNG', pnginfo=metadata)

                preview_images.append({
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "input"
                })

            # フロントエンドにプレビュー画像を送信
            try:
                preview_data = {
                    "id": unique_id,
                    "urls": preview_images
                }
                logging.info(f"MyNicheUtils: Sending preview data for node {unique_id}: {preview_data}")
                PromptServer.instance.send_sync("my-niche-utils-preview", preview_data)
                logging.info(f"MyNicheUtils: Successfully sent preview images for node {unique_id}")
            except Exception as e:
                logging.error(f"MyNicheUtils: Failed to send preview: {e}", exc_info=True)

            # auto_continueがTrueの場合は、ユーザー入力をスキップ
            if auto_continue:
                logging.info(f"MyNicheUtils: Auto continue enabled, skipping user input for node {unique_id}")
                # プレビューファイルをクリーンアップ
                try:
                    for preview_info in preview_images:
                        preview_path = os.path.join(input_dir, preview_info["filename"])
                        if os.path.exists(preview_path):
                            os.remove(preview_path)
                except Exception as e:
                    logging.warning(f"MyNicheUtils: Failed to cleanup preview files: {e}")

                # auto_continueの場合は元のマスクをそのまま返却（反転不要）
                return (image, mask_resized)

            # ユーザーからのメッセージを待機
            try:
                logging.info(f"MyNicheUtils: Waiting for user input for node {unique_id}")
                message = MessageHolder.waitForMessage(unique_id)
                logging.info(f"MyNicheUtils: Received message: {message}")

                # フロントエンドに処理完了を通知
                try:
                    PromptServer.instance.send_sync("my-niche-utils-complete", {
                        "id": unique_id,
                        "message": message
                    })
                    logging.info(f"MyNicheUtils: Sent completion signal for node {unique_id}")
                except Exception as e:
                    logging.error(f"MyNicheUtils: Failed to send completion signal: {e}")                # メッセージが"continue"の場合は処理を続行
                if message == "continue":
                    # MaskEditorが編集したファイルを探して読み込む
                    logging.info(f"MyNicheUtils: Looking for edited mask files...")

                    # clipspace ディレクトリを確認
                    input_dir = folder_paths.get_input_directory()
                    clipspace_dir = os.path.join(input_dir, "clipspace")

                    edited_mask = None
                    if os.path.exists(clipspace_dir):
                        # clipspace内の最新のpngファイルを探す
                        clipspace_files = [f for f in os.listdir(clipspace_dir) if f.endswith('.png')]
                        if clipspace_files:
                            # 最新のファイルを取得
                            latest_file = max(clipspace_files, key=lambda f: os.path.getmtime(os.path.join(clipspace_dir, f)))
                            latest_file_path = os.path.join(clipspace_dir, latest_file)

                            logging.info(f"MyNicheUtils: Found potential edited file: {latest_file}")

                            try:
                                # 編集されたファイルからマスクを抽出
                                with Image.open(latest_file_path) as edited_img:
                                    if 'A' in edited_img.getbands():
                                        # アルファチャンネルを取得
                                        alpha_channel = edited_img.getchannel('A')
                                        mask_array = np.array(alpha_channel).astype(np.float32) / 255.0

                                        # アルファチャンネルからComfyUIマスクへ変換
                                        # プレビュー時に 1.0 - alpha_mask でアルファチャンネルに保存したので、
                                        # 読み込み時は再度反転してComfyUIマスク形式に戻す
                                        # アルファチャンネル: 1.0が不透明（編集対象）、0.0が透明（編集対象外）
                                        # ComfyUIマスク: 1.0が編集対象外、0.0が編集対象 → 反転が必要
                                        mask_array = 1.0 - mask_array  # アルファチャンネルを反転してComfyUIマスクに変換
                                        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # [1, H, W]

                                        # バッチサイズに合わせて拡張
                                        if mask_tensor.shape[0] < batch_size:
                                            mask_tensor = mask_tensor.repeat(batch_size, 1, 1)

                                        edited_mask = mask_tensor
                                        logging.info(f"MyNicheUtils: Successfully extracted mask from edited file: {mask_tensor.shape}")
                                    else:
                                        logging.warning(f"MyNicheUtils: Edited file has no alpha channel")
                            except Exception as e:
                                logging.error(f"MyNicheUtils: Failed to process edited file: {e}")

                    if edited_mask is not None:
                        logging.info(f"MyNicheUtils: Using edited mask from MaskEditor")
                        # 元のプレビューファイルをクリーンアップ
                        try:
                            for preview_info in preview_images:
                                preview_path = os.path.join(input_dir, preview_info["filename"])
                                if os.path.exists(preview_path):
                                    os.remove(preview_path)
                        except Exception as e:
                            logging.warning(f"MyNicheUtils: Failed to cleanup preview files: {e}")

                        return (image, edited_mask)
                    else:
                        logging.warning(f"MyNicheUtils: No edited mask found, using original")
                        # 元のマスクをそのまま返却
                        return (image, mask_resized)
                else:
                    # キャンセルまたはその他のメッセージの場合
                    logging.info(f"MyNicheUtils: Operation cancelled or other message: {message}")
                    # 元のマスクをそのまま返却
                    return (image, mask_resized)

            except Exception as e:
                logging.error(f"MyNicheUtils: Error waiting for message: {e}")
                # エラー時も元のマスクをそのまま返却
                return (image, mask_resized)
        else:
            # プレビューが無効の場合（現在は使用されない）
            logging.info(f"MyNicheUtils: Preview disabled for node {unique_id}, returning immediately")

        logging.info(f"MyNicheUtils: MaskPainter execution completed for node {unique_id}")
        # 最後の処理でも元のマスクをそのまま返却
        return (image, mask_resized)


class MyNicheUtilsZipLoader:
    """
    ZipファイルまたはディレクトリからZipファイルを読み込んで、画像リストとして出力するノード
    """
    def __init__(self):
        self.last_dir_index = {}  # ディレクトリごとのインデックス管理
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["zip_file", "directory"],),
                "path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "zipファイルのパス、またはzipファイルが格納されたディレクトリのパス"
                }),
            },
            "optional": {
                "reset_index": ("BOOLEAN", {"default": False}),
                "reset_index_to": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "reset_indexがTrueの場合の開始インデックス（0ベース）"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(cls, mode, path, reset_index=False, reset_index_to=0, unique_id=None):
        if mode == "directory":
            # ディレクトリモードの場合は常に再実行されるように時間を返す
            import time
            return str(time.time())
        else:
            # zipファイルモードの場合はファイルのハッシュを返す
            if os.path.exists(path):
                m = hashlib.sha256()
                with open(path, 'rb') as f:
                    m.update(f.read())
                return m.digest().hex()
            return "not_found"

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGES", "ZIP_PATH", "IMAGE_COUNT", "CURRENT_INDEX", "TOTAL_FILES", "ZIP_FILENAME")
    FUNCTION = "load_zip"
    CATEGORY = "MyNicheUtils"

    def load_zip(self, mode, path, reset_index=False, reset_index_to=0, unique_id=None):
        if not path or path.strip() == "":
            raise ValueError("パスが指定されていません")

        zip_path = None
        current_index = 0
        total_files = 1

        if mode == "zip_file":
            # 直接zipファイル指定モード
            if not os.path.exists(path):
                raise FileNotFoundError(f"指定されたzipファイルが見つかりません: {path}")
            if not path.lower().endswith('.zip'):
                raise ValueError(f"zipファイルではありません: {path}")
            zip_path = path
            current_index = 1  # 単一ファイルの場合は1/1として表示
            total_files = 1

        elif mode == "directory":
            # ディレクトリ内zipファイル順次処理モード
            if not os.path.isdir(path):
                raise NotADirectoryError(f"指定されたパスがディレクトリではありません: {path}")

            # ディレクトリ内のzipファイルを辞書順で取得
            zip_pattern = os.path.join(path, "*.zip")
            zip_files = sorted(glob.glob(zip_pattern))

            if not zip_files:
                raise FileNotFoundError(f"指定されたディレクトリにzipファイルが見つかりません: {path}")

            total_files = len(zip_files)

            # インデックス管理
            if reset_index or unique_id not in self.last_dir_index:
                # reset_index_toが範囲外の場合は0に設定
                start_index = max(0, min(reset_index_to, total_files - 1))
                self.last_dir_index[unique_id] = start_index
                logging.info(f"MyNicheUtils ZipLoader: Reset index to {start_index} for node {unique_id}")

            current_index = self.last_dir_index[unique_id]

            # インデックスが範囲外の場合は最初に戻る
            if current_index >= len(zip_files):
                current_index = 0
                self.last_dir_index[unique_id] = 0

            zip_path = zip_files[current_index]

            # 次回のために次のインデックスを設定
            self.last_dir_index[unique_id] = (current_index + 1) % len(zip_files)

            # UIに表示するためのインデックスは1ベース
            display_index = current_index + 1

            logging.info(f"MyNicheUtils ZipLoader: Processing {zip_path} (index {display_index}/{total_files})")

        # zipファイルから画像を読み込み
        images = self._load_images_from_zip(zip_path)

        # zipファイル名を取得
        zip_filename = os.path.basename(zip_path)

        # ディレクトリモードの場合は1ベースのインデックスを返す
        if mode == "directory":
            return (images, zip_path, len(images), display_index, total_files, zip_filename)
        else:
            return (images, zip_path, len(images), current_index, total_files, zip_filename)

    def _load_images_from_zip(self, zip_path):
        """zipファイルから画像を読み込んでtensorのリストとして返す"""
        images = []

        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # zip内のファイル一覧を取得（辞書順でソート）
            file_list = sorted(zip_file.namelist())

            for filename in file_list:
                # ディレクトリは無視
                if filename.endswith('/'):
                    continue

                # 拡張子チェック
                _, ext = os.path.splitext(filename.lower())
                if ext not in self.supported_formats:
                    continue

                try:
                    # ファイルを読み込み
                    with zip_file.open(filename) as img_file:
                        img_data = img_file.read()

                    # PILで画像を開く
                    from io import BytesIO
                    img = Image.open(BytesIO(img_data))

                    # EXIFによる回転を適用
                    img = ImageOps.exif_transpose(img)

                    # アニメーション画像対応
                    for frame in ImageSequence.Iterator(img):
                        # RGB変換
                        if frame.mode == 'I':
                            frame = frame.point(lambda i: i * (1 / 255))
                        rgb_image = frame.convert("RGB")

                        # numpyに変換してtensorに
                        image_array = np.array(rgb_image).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_array)[None,]  # [1, H, W, C]
                        images.append(image_tensor)

                        # アニメーション画像の場合は最初のフレームのみ
                        if img.format not in ['GIF']:
                            break

                except Exception as e:
                    logging.warning(f"MyNicheUtils ZipLoader: Failed to load image {filename}: {e}")
                    continue

        if not images:
            logging.warning(f"MyNicheUtils ZipLoader: No valid images found in {zip_path}")
            # 空の画像を返す（エラーを避けるため）
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return empty_image

        # すべての画像を結合
        if len(images) == 1:
            return images[0]
        else:
            # バッチとして結合
            return torch.cat(images, dim=0)

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        mode = inputs.get("mode")
        path = inputs.get("path")

        if not path or path.strip() == "":
            return "パスが指定されていません"

        if mode == "zip_file":
            if not os.path.exists(path):
                return f"指定されたzipファイルが見つかりません: {path}"
            if not path.lower().endswith('.zip'):
                return f"zipファイルではありません: {path}"
        elif mode == "directory":
            if not os.path.exists(path):
                return f"指定されたパスが存在しません: {path}"
            if not os.path.isdir(path):
                return f"指定されたパスがディレクトリではありません: {path}"

            # ディレクトリ内にzipファイルが存在するかチェック
            zip_pattern = os.path.join(path, "*.zip")
            zip_files = glob.glob(zip_pattern)
            if not zip_files:
                return f"指定されたディレクトリにzipファイルが見つかりません: {path}"

        return True

class MyNicheUtilsImageLoader:
    """
    ディレクトリから画像ファイルを読み込んで、画像リストとして出力するノード
    """
    def __init__(self):
        self.last_dir_index = {}  # ディレクトリごとのインデックス管理
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["directory", "parent_directory"],),
                "path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "画像が格納されたディレクトリのパス、または複数ディレクトリが格納された親ディレクトリのパス"
                }),
            },
            "optional": {
                "reset_index": ("BOOLEAN", {"default": False}),
                "reset_index_to": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "reset_indexがTrueの場合の開始インデックス（0ベース）"
                }),
                "recursive": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "directoryモードでサブディレクトリも再帰的に検索するかどうか"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def IS_CHANGED(cls, mode, path, reset_index=False, reset_index_to=0, recursive=False, unique_id=None):
        # ディレクトリ内の画像ファイル数や更新時刻が変わった場合に再実行されるように時間を返す
        import time
        return str(time.time())

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGES", "DIR_PATH", "IMAGE_COUNT", "CURRENT_INDEX", "TOTAL_DIRS", "DIR_NAME")
    FUNCTION = "load_images"
    CATEGORY = "MyNicheUtils"

    def load_images(self, mode, path, reset_index=False, reset_index_to=0, recursive=False, unique_id=None):
        if not path or path.strip() == "":
            raise ValueError("パスが指定されていません")

        dir_path = None
        current_index = 0
        total_dirs = 1

        if mode == "directory":
            # 直接ディレクトリ指定モード
            if not os.path.exists(path):
                raise FileNotFoundError(f"指定されたディレクトリが見つかりません: {path}")
            if not os.path.isdir(path):
                raise NotADirectoryError(f"指定されたパスがディレクトリではありません: {path}")

            dir_path = path
            current_index = 1  # 単一ディレクトリの場合は1/1として表示
            total_dirs = 1

        elif mode == "parent_directory":
            # 親ディレクトリ内のサブディレクトリを順次処理モード
            if not os.path.isdir(path):
                raise NotADirectoryError(f"指定されたパスがディレクトリではありません: {path}")

            # 親ディレクトリ内のサブディレクトリを辞書順で取得
            subdirs = [d for d in os.listdir(path)
                      if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')
                      and not d == "clipspace"]  # clipspaceは除外
            subdirs = sorted(subdirs)

            if not subdirs:
                raise FileNotFoundError(f"指定されたディレクトリにサブディレクトリが見つかりません: {path}")

            total_dirs = len(subdirs)

            # インデックス管理
            if reset_index or unique_id not in self.last_dir_index:
                # reset_index_toが範囲外の場合は0に設定
                start_index = max(0, min(reset_index_to, total_dirs - 1))
                self.last_dir_index[unique_id] = start_index
                logging.info(f"MyNicheUtils ImageLoader: Reset index to {start_index} for node {unique_id}")

            current_index = self.last_dir_index[unique_id]

            # インデックスが範囲外の場合は最初に戻る
            if current_index >= len(subdirs):
                current_index = 0
                self.last_dir_index[unique_id] = 0

            dir_path = os.path.join(path, subdirs[current_index])

            # 次回のために次のインデックスを設定
            self.last_dir_index[unique_id] = (current_index + 1) % len(subdirs)

            # UIに表示するためのインデックスは1ベース
            display_index = current_index + 1

            logging.info(f"MyNicheUtils ImageLoader: Processing {dir_path} (index {display_index}/{total_dirs})")

        # ディレクトリから画像を読み込み
        images = self._load_images_from_directory(dir_path, recursive)

        # ディレクトリ名を取得
        dir_name = os.path.basename(dir_path)

        # parent_directoryモードの場合は1ベースのインデックスを返す
        if mode == "parent_directory":
            return (images, dir_path, len(images), display_index, total_dirs, dir_name)
        else:
            return (images, dir_path, len(images), current_index, total_dirs, dir_name)

    def _load_images_from_directory(self, dir_path, recursive=False):
        """ディレクトリから画像を読み込んでtensorのリストとして返す"""
        images = []
        image_files = []

        if recursive:
            # 再帰的に画像ファイルを取得
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file.lower())
                    if ext in self.supported_formats:
                        image_files.append(file_path)
        else:
            # 直接の子ファイルのみ取得
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file.lower())
                    if ext in self.supported_formats:
                        image_files.append(file_path)

        # ファイルパスでソート
        image_files = sorted(image_files)

        for file_path in image_files:
            try:
                # PILで画像を開く
                img = Image.open(file_path)

                # EXIFによる回転を適用
                img = ImageOps.exif_transpose(img)

                # アニメーション画像対応
                for frame in ImageSequence.Iterator(img):
                    # RGB変換
                    if frame.mode == 'I':
                        frame = frame.point(lambda i: i * (1 / 255))
                    rgb_image = frame.convert("RGB")

                    # numpyに変換してtensorに
                    image_array = np.array(rgb_image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_array)[None,]  # [1, H, W, C]
                    images.append(image_tensor)

                    # アニメーション画像の場合は最初のフレームのみ
                    if img.format not in ['GIF']:
                        break

            except Exception as e:
                logging.warning(f"MyNicheUtils ImageLoader: Failed to load image {file_path}: {e}")
                continue

        if not images:
            logging.warning(f"MyNicheUtils ImageLoader: No valid images found in {dir_path}")
            # 空の画像を返す（エラーを避けるため）
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return empty_image

        # すべての画像を結合
        if len(images) == 1:
            return images[0]
        else:
            # バッチとして結合
            return torch.cat(images, dim=0)

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        mode = inputs.get("mode")
        path = inputs.get("path")

        if not path or path.strip() == "":
            return "パスが指定されていません"

        if not os.path.exists(path):
            return f"指定されたパスが存在しません: {path}"

        if not os.path.isdir(path):
            return f"指定されたパスがディレクトリではありません: {path}"

        if mode == "parent_directory":
            # 親ディレクトリ内にサブディレクトリが存在するかチェック
            subdirs = [d for d in os.listdir(path)
                      if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')]
            if not subdirs:
                return f"指定されたディレクトリにサブディレクトリが見つかりません: {path}"

        return True


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MyNicheUtilsSaveImage": MyNicheUtilsSaveImage,
    "MyNicheUtilsLoadImage": MyNicheUtilsLoadImage,
    "MyNicheUtilsImageBatch": MyNicheUtilsImageBatch,
    "MyNicheUtilsCounter": MyNicheUtilsCounter,
    "MyNicheUtilsMaskPainter": MyNicheUtilsMaskPainter,
    "MyNicheUtilsZipLoader": MyNicheUtilsZipLoader,
    "MyNicheUtilsImageLoader": MyNicheUtilsImageLoader,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNicheUtilsSaveImage": "SaveImage Node",
    "MyNicheUtilsLoadImage": "LoadImage Node",
    "MyNicheUtilsImageBatch": "ImageBatch Node",
    "MyNicheUtilsCounter": "Counter Node",
    "MyNicheUtilsMaskPainter": "MaskPainter Node",
    "MyNicheUtilsZipLoader": "ZipLoader Node",
    "MyNicheUtilsImageLoader": "ImageLoader Node",
}