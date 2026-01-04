import sys
import os
import glob

# --- macOS: Qt platform plugin (cocoa) の自動検出 ---
# VSCode/venv 環境で「Could not find the Qt platform plugin 'cocoa'」が出る場合があるため、
# PyQt5 の plugins/platforms を見つけて環境変数に設定する（Qt import より前に行う）。
if sys.platform == "darwin":
    try:
        # 代表的な場所を候補として探索
        candidates = []
        # venv / system python の site-packages
        candidates += glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "PyQt5", "Qt5", "plugins"))
        candidates += glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "PyQt5", "Qt", "plugins"))
        # Homebrew python などで異なる配置の場合
        candidates += glob.glob(os.path.join(sys.prefix, "..", "lib", "python*", "site-packages", "PyQt5", "Qt5", "plugins"))

        plugin_root = next((p for p in candidates if os.path.isdir(p)), None)
        if plugin_root:
            # platforms に libqcocoa.dylib がある
            platforms_dir = os.path.join(plugin_root, "platforms")
            if os.path.isdir(platforms_dir):
                os.environ.setdefault("QT_PLUGIN_PATH", plugin_root)
                os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", platforms_dir)
    except Exception:
        pass
# -----------------------------------------------

import cv2
import torch
from ultralytics import YOLO
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5 import QtMultimedia

# ==== 推論設定（精度重視のプリセット）====
MODEL_WEIGHTS = "yolo12m.pt"   
IMG_SIZE = 1280                 
CONF_THRES = 0.3
IOU_THRES = 0.45
# ========================================

# 使用デバイスを自動選択（MPS対応MacならGPUを使う）
device = "mps" if torch.backends.mps.is_available() else "cpu"

class YoloLiveWindow(QtWidgets.QMainWindow):
    def resizeEvent(self, event):
        """ウィンドウサイズ変更時にプレビューを滑らかに再スケールする。"""
        try:
            pm = self.label.pixmap()
            if pm is not None and not pm.isNull():
                scaled = pm.scaled(
                    self.label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.label.setPixmap(scaled)
        except Exception:
            pass
        super().resizeEvent(event)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ひと検知システム")
        self._rgb_ref = None  # QImage が参照するバッファを保持（破棄による表示崩れ防止）

        # ウィンドウボタン: 閉じる(赤)・ズーム(緑)のみを有効化（最小化=黄を無効）
        flags = (
            QtCore.Qt.Window
            | QtCore.Qt.WindowTitleHint
            | QtCore.Qt.WindowCloseButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint  # mac の緑ボタン
        )
        self.setWindowFlags(flags)

        # 表示領域
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.resize(960, 540)
        # 画像を小さくできるように（最小サイズとサイズポリシーを緩くする）
        self.setMinimumSize(320, 180)
        self.label.setMinimumSize(1, 1)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)


        # カメラ
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "エラー", "カメラが見つかりません。接続を確認してください。")
            QtCore.QTimer.singleShot(0, self.close)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        cv2.setUseOptimized(True)

        # モデル（単一モデル）
        self.model = None
        try:
            self.model = YOLO(MODEL_WEIGHTS)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "モデル読み込みエラー", f"{MODEL_WEIGHTS} を読み込めませんでした。\n{e}")
            QtCore.QTimer.singleShot(0, self.close)
            return

        # COCOクラス名の上書き（0: person → 人）
        try:
            self.model.model.names[0] = "人"
        except Exception:
            pass

        # タイマーで周期的に更新（約30FPS目安）
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 30FPS目安（33ms間隔）

        # 人数の増加（0→1 を含む）で鳴動するためのカウンタ
        self.prev_person_count = 0
        self.alert_active = False


        # フォールバック用ビープのタイマー（200ms間隔で連続ビープ）
        self.beep_timer = QtCore.QTimer(self)
        self.beep_timer.setInterval(200)
        self.beep_timer.timeout.connect(lambda: QtWidgets.QApplication.beep())

        # 警告停止用（5秒後に止める）
        self.stop_sound_timer = QtCore.QTimer(self)
        self.stop_sound_timer.setSingleShot(True)
        self.stop_sound_timer.timeout.connect(self._stop_warning_sound)

        # 合成トーン再生の準備（QtMultimedia、外部ファイルは使わない）
        self.audio_output = None
        self.audio_buffer = None
        self._prepare_tone_audio()

    def _infer(self, frame):
        """推論の安全実行（失敗時はダミーを返す）。"""
        class _Dummy:
            def __init__(self, img):
                self._img = img
                self.boxes = None
            def plot(self, conf=False):
                return self._img
        if self.model is None:
            return [_Dummy(frame)]
        try:
            return self.model(
                frame,
                device=device,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                iou=IOU_THRES,
                agnostic_nms=False,
                classes=[0],  # person
                max_det=100,
                augment=False,
                verbose=False,
            )
        except Exception:
            return [_Dummy(frame)]

    def update_frame(self):
        if not self.cap:
            return
        # フレーム取得（最大3回リトライ）
        retry_count = 0
        ok, frame = self.cap.read()
        while not ok and retry_count < 3:
            retry_count += 1
            QtCore.QThread.msleep(200)  # 0.2秒待機して再試行
            ok, frame = self.cap.read()

        if not ok:
            self.timer.stop()
            QtWidgets.QMessageBox.warning(self, "警告", "フレームを3回試みましたが取得できません。終了します。")
            self.close()
            return


        # 推論（personのみ）
        results = self._infer(frame)

        # まず高信頼度でフィルタ→人数カウント→警告判定（表示と一致させる）
        try:
            if hasattr(results[0], "boxes") and results[0].boxes is not None:
                boxes = results[0].boxes
                if hasattr(boxes, "conf") and boxes.conf is not None:
                    keep_mask = boxes.conf >= 0.56
                    # 1件以上あればフィルタ適用（全件 True のときも OK）
                    if hasattr(keep_mask, "numel") and keep_mask.numel() > 0:
                        results[0].boxes = boxes[keep_mask]
        except Exception:
            pass

        # 人数が前回より増えたとき（0→1 を含む）に警告音を鳴らす（フィルタ後の数で判定）
        try:
            boxes = results[0].boxes if hasattr(results[0], "boxes") else None
            current_count = len(boxes) if boxes is not None else 0
            if current_count > self.prev_person_count:
                self._start_warning_sound(3000)  # 3秒
            self.prev_person_count = current_count
        except Exception:
            pass

        try:
            annotated = results[0].plot(conf=False)
        except Exception:
            annotated = frame
        try:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = annotated

        # QImage は numpy バッファを参照するため、参照切れを防ぐ
        self._rgb_ref = rgb

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # ← OpenCV は日本語描画非対応のため、Qtで人数を描画（白背景）
        try:
            # 直前の検出結果から人数を再計算
            boxes2 = results[0].boxes if hasattr(results[0], "boxes") else None
            count_display = len(boxes2) if boxes2 is not None else 0

            text_lines = [
                f"人数: {count_display}",
            ]

            painter = QtGui.QPainter(qimg)
            font = QtGui.QFont()
            font.setPointSize(24)          # 見やすいサイズ
            font.setBold(True)
            painter.setFont(font)

            metrics = QtGui.QFontMetrics(font)
            line_height = metrics.height()
            max_width = max(metrics.horizontalAdvance(t) for t in text_lines)

            x = 12
            y_start = 12 + line_height

            # 白背景の矩形（2行ぶん）
            bg_rect = QtCore.QRect(
                x - 8,
                y_start - line_height - 8,
                max_width + 16,
                line_height * len(text_lines) + 8,
            )
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 255, 255))
            painter.drawRect(bg_rect)

            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
            for i, t in enumerate(text_lines):
                y = y_start + i * line_height
                painter.drawText(x, y - metrics.descent() // 2, t)

            painter.end()
        except Exception:
            pass

        pixmap = QtGui.QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)


    def _prepare_tone_audio(self):
        """QtMultimedia で合成トーンを再生できるよう準備（外部ファイルは使わない）。"""
        try:
            fmt = QtMultimedia.QAudioFormat()
            fmt.setSampleRate(22050)  # 負荷を下げるためにサンプルレートを下げる
            fmt.setChannelCount(1)
            fmt.setSampleSize(16)
            fmt.setCodec("audio/pcm")
            fmt.setByteOrder(QtMultimedia.QAudioFormat.LittleEndian)
            fmt.setSampleType(QtMultimedia.QAudioFormat.SignedInt)

            info = QtMultimedia.QAudioDeviceInfo.defaultOutputDevice()
            if not info.isFormatSupported(fmt):
                # デバイスの推奨フォーマットへフォールバック
                pf = info.preferredFormat()
                fmt = pf
                if not info.isFormatSupported(fmt):
                    self.audio_output = None
                    return

            self.audio_output = QtMultimedia.QAudioOutput(info, fmt, self)
            # 大きめの内部バッファでドロップ対策（約5秒分）
            sr = fmt.sampleRate()
            channels = fmt.channelCount()
            bytes_per_sample = fmt.sampleSize() // 8
            desired_bytes = sr * channels * bytes_per_sample * 5  # 5秒分
            try:
                self.audio_output.setBufferSize(desired_bytes)
            except Exception:
                pass
            # 音量を少し大きめに（0.0〜1.0）
            self.audio_output.setVolume(1.0)  # できるだけ大きく
        except Exception:
            self.audio_output = None

    def _generate_tone_buffer(self, duration_ms=5000, freq=880):
        """指定時間のサイン波PCMを生成し、QBuffer を返す。"""
        try:
            fmt = self.audio_output.format()
            sr = fmt.sampleRate()
            samples = int(sr * duration_ms / 1000)

            import math, array
            amplitude = int(32767 * 0.85)
            data = array.array('h', (0 for _ in range(samples)))
            for n in range(samples):
                data[n] = int(amplitude * math.sin(2 * math.pi * freq * n / sr))

            ba = QtCore.QByteArray(data.tobytes())
            buf = QtCore.QBuffer(self)
            buf.setData(ba)
            buf.open(QtCore.QIODevice.ReadOnly)
            return buf
        except Exception:
            return None

    def _start_warning_sound(self, duration_ms=5000):
        """5秒間の警告音を開始（合成トーン優先、失敗時はビープ連打）。"""
        if self.alert_active:
            return
        self.alert_active = True

        used_multimedia = False
        try:
            if self.audio_output is None:
                self._prepare_tone_audio()
            if self.audio_output is not None:
                self.audio_buffer = self._generate_tone_buffer(duration_ms)
                if self.audio_buffer is not None:
                    self.audio_output.start(self.audio_buffer)
                    used_multimedia = True
        except Exception:
            used_multimedia = False

        if not used_multimedia:
            # QtMultimedia が使えない環境では 200ms 間隔でビープを鳴らす
            self.beep_timer.start()

        # 所定時間で停止
        self.stop_sound_timer.start(duration_ms)

    def _stop_warning_sound(self):
        """警告音の停止と後始末。"""
        try:
            if self.audio_output is not None:
                self.audio_output.stop()
            if self.audio_buffer is not None:
                self.audio_buffer.close()
                self.audio_buffer = None
            if self.beep_timer.isActive():
                self.beep_timer.stop()
        except Exception:
            pass
        self.alert_active = False

    def closeEvent(self, event):
        # 終了処理
        try:
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "cap") and self.cap:
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = YoloLiveWindow()
    win.show()
    sys.exit(app.exec_())