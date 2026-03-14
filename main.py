
from kivy.app import App
from kivy.uix.image import Image as KivyImage
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.audio import SoundLoader
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.modalview import ModalView
from kivy.animation import Animation
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.graphics import Color, Rectangle, Line
import sqlite3
import os
from datetime import datetime
import numpy as np
import cv2
import face_recognition
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont, ImageEnhance
import arabic_reshaper
import time
import sys
import math
import pickle

def reshape_arabic(text):
    if not text or not isinstance(text, str):
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return reshaped[::-1]
    except:
        return text[::-1]

def draw_arabic_text(image, text, position, font_path, font_size=32, color=(0, 255, 0)):
    try:
        img_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        reshaped_text = arabic_reshaper.reshape(text)
        display_text = reshaped_text[::-1]
        draw.text(position, display_text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"خطأ في رسم النص العربي: {str(e)}")
        return image

def enhance_face_for_recognition(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(image, 0.5, enhanced_bgr, 0.5, 0)
        return blended
    except Exception as e:
        print(f"خطأ في تحسين الصورة: {str(e)}")
        return image

def preprocess_image_for_recognition(image):
    try:
        h, w = image.shape[:2]
        if h > 800 or w > 800:
            scale = 800 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return image
    except Exception as e:
        print(f"خطأ في معالجة الصورة: {str(e)}")
        return image

def detect_face_with_enhancement(image):
    try:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        if face_locations:
            return face_locations
        enhanced = enhance_face_for_recognition(image)
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_enhanced, model='hog')
        if face_locations:
            return face_locations
        h, w = image.shape[:2]
        scaled = cv2.resize(image, (w*2, h*2))
        rgb_scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_scaled, model='hog')
        if face_locations:
            adjusted_locations = []
            for (top, right, bottom, left) in face_locations:
                adjusted_locations.append((top//2, right//2, bottom//2, left//2))
            return adjusted_locations
        return []
    except Exception as e:
        print(f"خطأ في الكشف عن الوجه: {str(e)}")
        return []

class ArabicTextInput(BoxLayout):
    def __init__(self, label_text='', hint_text='', **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = 160
        self.spacing = 5
        self.padding = [0, 0, 0, 0]
        self.field_label = Label(
            text=reshape_arabic(label_text + ':'),
            font_name='/storage/emulated/0/Amiri-Regular.ttf',
            font_size=40,
            size_hint_y=0.3,
            halign='right',
            valign='middle',
            color=(0.2, 0.4, 0.8, 1),
            bold=True
        )
        self.field_label.bind(size=self.field_label.setter('text_size'))
        self.add_widget(self.field_label)
        self.text_input = TextInput(
            multiline=False,
            hint_text=reshape_arabic(hint_text),
            font_size=42,
            font_name='/storage/emulated/0/Amiri-Regular.ttf',
            padding=[15, 8],
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            cursor_color=(0.2, 0.6, 1, 1),
            halign='right',
            size_hint_y=0.4
        )
        self.add_widget(self.text_input)
        self.preview_label = Label(
            text='',
            font_name='/storage/emulated/0/Amiri-Regular.ttf',
            font_size=38,
            size_hint_y=0.3,
            halign='right',
            valign='middle',
            color=(0.2, 0.8, 0.2, 1),
            italic=True
        )
        self.preview_label.bind(size=self.preview_label.setter('text_size'))
        self.add_widget(self.preview_label)
        self.text_input.bind(text=self.on_text_change)
        self.original_text = ''
    
    def on_text_change(self, instance, value):
        self.original_text = value
        if value.strip():
            self.preview_label.text = reshape_arabic(value)
        else:
            self.preview_label.text = ''
    
    def get_text(self):
        return self.original_text
    
    def set_text(self, text):
        self.text_input.text = text
        self.original_text = text
        self.preview_label.text = reshape_arabic(text) if text else ''

class AnimatedButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0.2, 0.6, 1, 1)
        self.font_name = '/storage/emulated/0/Amiri-Regular.ttf'
        
    def on_press(self):
        anim = Animation(background_color=(0.1, 0.3, 0.5, 1), duration=0.1)
        anim.start(self)
        return super().on_press()
    
    def on_release(self):
        anim = Animation(background_color=(0.2, 0.6, 1, 1), duration=0.1)
        anim.start(self)
        return super().on_release()

class ArabicPopup(ModalView):
    def __init__(self, title, message, popup_type='info', **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.85, 0.5)
        self.auto_dismiss = False
        self.background_color = (0, 0, 0, 0.9)
        if popup_type == 'success':
            title_color = (0.2, 0.8, 0.2, 1)
        elif popup_type == 'error':
            title_color = (0.8, 0.2, 0.2, 1)
        elif popup_type == 'warning':
            title_color = (1, 0.6, 0, 1)
        else:
            title_color = (0.2, 0.6, 1, 1)
        layout = BoxLayout(orientation='vertical', padding=25, spacing=20)
        title_label = Label(
            text=reshape_arabic(title),
            font_name='/storage/emulated/0/Amiri-Regular.ttf',
            font_size=55,
            size_hint=(1, 0.25),
            halign='center',
            valign='middle',
            color=title_color,
            bold=True
        )
        title_label.bind(size=title_label.setter('text_size'))
        message_label = Label(
            text=reshape_arabic(message),
            font_name='/storage/emulated/0/Amiri-Regular.ttf',
            font_size=45,
            size_hint=(1, 0.5),
            halign='center',
            valign='middle',
            color=(1, 1, 1, 1)
        )
        message_label.bind(size=message_label.setter('text_size'))
        btn_close = AnimatedButton(
            text=reshape_arabic('موافق'),
            font_size=50,
            background_color=title_color,
            color=(1, 1, 1, 1),
            size_hint=(1, 0.2)
        )
        btn_close.bind(on_press=self.dismiss)
        layout.add_widget(title_label)
        layout.add_widget(message_label)
        layout.add_widget(btn_close)
        self.add_widget(layout)

class MultiFaceCapture(ModalView):
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.95, 0.95)
        self.auto_dismiss = False
        self.background_color = (0, 0, 0, 0.95)
        self.app = app_instance
        self.captured_images = []
        self.current_angle = 0
        self.angles = ['أمامي', 'يمين 45°', 'يسار 45°', 'أعلى', 'أسفل']
        self.last_capture_time = 0
        self.capture_cooldown = 1
        self.camera_texture = None
        self.build_ui()
        self.start_camera()
    
    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        title = Label(
            text=reshape_arabic('📸 التقاط صور متعددة للوجه'),
            font_name=self.app.font_path,
            font_size=55,
            size_hint_y=0.08,
            bold=True,
            color=(0.2, 0.8, 1, 1)
        )
        main_layout.add_widget(title)
        self.angle_label = Label(
            text=reshape_arabic(f'الزاوية: {self.angles[0]}'),
            font_name=self.app.font_path,
            font_size=45,
            size_hint_y=0.05,
            color=(1, 1, 0, 1)
        )
        main_layout.add_widget(self.angle_label)
        self.camera_layout = BoxLayout(size_hint_y=0.5, padding=5)
        with self.camera_layout.canvas.before:
            Color(0.2, 0.2, 0.2, 1)
            self.cam_bg_rect = Rectangle(pos=self.camera_layout.pos, size=self.camera_layout.size)
        def update_cam_bg(instance, value):
            self.cam_bg_rect.pos = instance.pos
            self.cam_bg_rect.size = instance.size
        self.camera_layout.bind(pos=update_cam_bg, size=update_cam_bg)
        from kivy.uix.camera import Camera
        self.camera = Camera(play=True, resolution=(640, 480))
        self.camera.allow_stretch = True
        self.camera.keep_ratio = False
        self.camera_layout.add_widget(self.camera)
        main_layout.add_widget(self.camera_layout)
        self.guide_frame = Label(
            text='👤',
            font_size=200,
            color=(1, 1, 1, 0.3),
            size_hint=(None, None),
            size=(200, 200),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        self.camera_layout.add_widget(self.guide_frame)
        controls_layout = BoxLayout(size_hint_y=0.1, spacing=10, padding=10)
        btn_capture = AnimatedButton(
            text=reshape_arabic('📸 التقاط صورة'),
            font_size=35,
            background_color=(0.2, 0.7, 0.3, 1),
            size_hint_x=0.33
        )
        btn_capture.bind(on_press=self.capture_image)
        btn_next = AnimatedButton(
            text=reshape_arabic('⏭️ التالي'),
            font_size=35,
            background_color=(0.2, 0.6, 1, 1),
            size_hint_x=0.33
        )
        btn_next.bind(on_press=self.next_angle)
        btn_finish = AnimatedButton(
            text=reshape_arabic('✅ إنهاء'),
            font_size=35,
            background_color=(0.5, 0.5, 0.5, 1),
            size_hint_x=0.33
        )
        btn_finish.bind(on_press=self.finish_capture)
        controls_layout.add_widget(btn_capture)
        controls_layout.add_widget(btn_next)
        controls_layout.add_widget(btn_finish)
        main_layout.add_widget(controls_layout)
        preview_layout = BoxLayout(size_hint_y=0.2, spacing=5, padding=5)
        preview_label = Label(
            text=reshape_arabic('الصور الملتقطة:'),
            font_name=self.app.font_path,
            font_size=35,
            size_hint_x=0.2
        )
        preview_layout.add_widget(preview_label)
        self.images_preview = BoxLayout(size_hint_x=0.8, spacing=5)
        preview_layout.add_widget(self.images_preview)
        main_layout.add_widget(preview_layout)
        btn_cancel = AnimatedButton(
            text=reshape_arabic('❌ إلغاء'),
            font_size=40,
            background_color=(0.8, 0.2, 0.2, 1),
            size_hint_y=0.07
        )
        btn_cancel.bind(on_press=self.cancel)
        main_layout.add_widget(btn_cancel)
        self.add_widget(main_layout)
    
    def start_camera(self):
        Clock.schedule_interval(self.update_camera, 1/30)
    
    def update_camera(self, dt):
        if not self.camera.texture:
            return
        try:
            texture = self.camera.texture
            size = texture.size
            pixels = texture.pixels
            frame = np.frombuffer(pixels, dtype=np.uint8)
            frame = frame.reshape(size[1], size[0], 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_flip = np.flip(frame_rgb, 0)
            texture = Texture.create(size=(frame_flip.shape[1], frame_flip.shape[0]), colorfmt='rgba')
            texture.blit_buffer(frame_flip.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
            self.camera.texture = texture
            self.camera_texture = frame.copy()
        except Exception as e:
            print(f"خطأ في تحديث الكاميرا: {str(e)}")
    
    def capture_image(self, instance):
        current_time = time.time()
        if current_time - self.last_capture_time < self.capture_cooldown:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ انتظر قليلاً قبل التقاط صورة أخرى',
                popup_type='warning'
            )
            popup.open()
            return
        if self.camera_texture is None:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ الكاميرا غير جاهزة بعد',
                popup_type='warning'
            )
            popup.open()
            return
        try:
            frame = self.camera_texture.copy()
            face_locations = detect_face_with_enhancement(frame)
            if not face_locations:
                popup = ArabicPopup(
                    title='تنبيه',
                    message='❌ لم يتم اكتشاف وجه!\nتأكد من:\n• وضع الوجه في الإطار\n• الإضاءة المناسبة\n• المسافة المناسبة',
                    popup_type='error'
                )
                popup.open()
                return
            self.captured_images.append(frame.copy())
            self.last_capture_time = current_time
            self.update_preview()
            popup = ArabicPopup(
                title='✅ تم الالتقاط',
                message=f'تم التقاط الصورة {len(self.captured_images)} بنجاح',
                popup_type='success'
            )
            popup.open()
        except Exception as e:
            print(f"خطأ في التقاط الصورة: {str(e)}")
            popup = ArabicPopup(
                title='خطأ',
                message='❌ حدث خطأ في التقاط الصورة',
                popup_type='error'
            )
            popup.open()
    
    def update_preview(self):
        self.images_preview.clear_widgets()
        for i, img in enumerate(self.captured_images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (80, 80))
            img_widget = KivyImage(
                size_hint=(None, 1),
                width=80,
                allow_stretch=True,
                keep_ratio=True
            )
            img_flip = np.flip(img_resized, 0)
            texture = Texture.create(size=(80, 80), colorfmt='rgb')
            texture.blit_buffer(img_flip.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            img_widget.texture = texture
            img_container = BoxLayout(orientation='vertical', size_hint=(None, 1), width=80)
            img_container.add_widget(img_widget)
            img_number = Label(
                text=str(i+1),
                font_name=self.app.font_path,
                font_size=25,
                size_hint_y=0.2,
                color=(1, 1, 0, 1)
            )
            img_container.add_widget(img_number)
            self.images_preview.add_widget(img_container)
    
    def next_angle(self, instance):
        if len(self.captured_images) == 0:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ يرجى التقاط صورة أولاً',
                popup_type='warning'
            )
            popup.open()
            return
        self.current_angle += 1
        if self.current_angle < len(self.angles):
            self.angle_label.text = reshape_arabic(f'الزاوية: {self.angles[self.current_angle]}')
            message = f"""
📸 التقط صورة من الزاوية: {self.angles[self.current_angle]}
✅ تم التقاط {len(self.captured_images)} صور
"""
            popup = ArabicPopup(
                title='الزاوية التالية',
                message=message,
                popup_type='info'
            )
            popup.open()
    
    def finish_capture(self, instance):
        if len(self.captured_images) == 0:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ يرجى التقاط صورة واحدة على الأقل',
                popup_type='warning'
            )
            popup.open()
            return
        message = f"""
✅ تم التقاط {len(self.captured_images)} صور بنجاح
سيتم استخدام هذه الصور للتسجيل
"""
        popup = ArabicPopup(
            title='🎉 اكتمل التصوير',
            message=message,
            popup_type='success'
        )
        popup.bind(on_dismiss=self.go_to_registration)
        popup.open()
    
    def go_to_registration(self, instance):
        self.app.multi_face_images = self.captured_images
        self.dismiss()
        self.app.show_registration_form()
    
    def cancel(self, instance):
        popup = ModalView(size_hint=(0.8, 0.4), auto_dismiss=False)
        popup.background_color = (0, 0, 0, 0.9)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        title = Label(
            text=reshape_arabic('تأكيد الإلغاء'),
            font_name=self.app.font_path,
            font_size=45,
            color=(1, 0.6, 0, 1),
            size_hint_y=0.3
        )
        layout.add_widget(title)
        message = Label(
            text=reshape_arabic('هل تريد إلغاء عملية التسجيل؟'),
            font_name=self.app.font_path,
            font_size=35,
            color=(1, 1, 1, 1),
            size_hint_y=0.4
        )
        layout.add_widget(message)
        buttons = BoxLayout(size_hint_y=0.3, spacing=10)
        btn_yes = AnimatedButton(
            text=reshape_arabic('✅ نعم'),
            font_size=35,
            background_color=(0.2, 0.7, 0.3, 1),
            size_hint_x=0.5
        )
        btn_yes.bind(on_press=lambda x: self.confirm_cancel(popup))
        btn_no = AnimatedButton(
            text=reshape_arabic('❌ لا'),
            font_size=35,
            background_color=(0.8, 0.2, 0.2, 1),
            size_hint_x=0.5
        )
        btn_no.bind(on_press=popup.dismiss)
        buttons.add_widget(btn_yes)
        buttons.add_widget(btn_no)
        layout.add_widget(buttons)
        popup.add_widget(layout)
        popup.open()
    
    def confirm_cancel(self, popup):
        popup.dismiss()
        self.dismiss()
        self.app.freeze_frame = False
        self.app.camera.play = True

class FolderSelectionView(ModalView):
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.95, 0.95)
        self.auto_dismiss = False
        self.background_color = (0, 0, 0, 0.95)
        self.app = app_instance
        self.current_path = "/storage/emulated/0/صور"
        self.selected_folder = None
        self.build_ui()
    
    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        title = Label(
            text=reshape_arabic('📁 اختيار مجلد الصور'),
            font_name=self.app.font_path,
            font_size=55,
            size_hint_y=0.08,
            bold=True,
            color=(0.2, 0.8, 1, 1)
        )
        main_layout.add_widget(title)
        self.path_label = Label(
            text=reshape_arabic(f'📂 {self.current_path}'),
            font_name=self.app.font_path,
            font_size=35,
            size_hint_y=0.05,
            color=(1, 1, 0, 1),
            shorten=True,
            shorten_from='right'
        )
        main_layout.add_widget(self.path_label)
        nav_layout = BoxLayout(size_hint_y=0.08, spacing=5, padding=5)
        btn_back = AnimatedButton(
            text=reshape_arabic('🔙 رجوع'),
            font_size=30,
            background_color=(0.5, 0.5, 0.5, 1),
            size_hint_x=0.5
        )
        btn_back.bind(on_press=self.go_back)
        btn_root = AnimatedButton(
            text=reshape_arabic('🏠 الرئيسي'),
            font_size=30,
            background_color=(0.2, 0.6, 1, 1),
            size_hint_x=0.5
        )
        btn_root.bind(on_press=self.go_to_root)
        nav_layout.add_widget(btn_back)
        nav_layout.add_widget(btn_root)
        main_layout.add_widget(nav_layout)
        scroll_layout = ScrollView(size_hint_y=0.1)
        self.folders_grid = GridLayout(cols=2, spacing=10, padding=10, size_hint_y=None)
        self.folders_grid.bind(minimum_height=self.folders_grid.setter('height'))
        self.load_folders()
        scroll_layout.add_widget(self.folders_grid)
        main_layout.add_widget(scroll_layout)
        controls_layout = BoxLayout(size_hint_y=0.15, spacing=10, padding=10, orientation='vertical')
        btn_select = AnimatedButton(
            text=reshape_arabic('✅ اختيار هذا المجلد'),
            font_size=40,
            background_color=(0.2, 0.7, 0.3, 1),
            size_hint_y=0.5
        )
        btn_select.bind(on_press=self.select_folder)
        btn_cancel = AnimatedButton(
            text=reshape_arabic('❌ إلغاء'),
            font_size=40,
            background_color=(0.8, 0.2, 0.2, 1),
            size_hint_y=0.5
        )
        btn_cancel.bind(on_press=self.cancel)
        controls_layout.add_widget(btn_select)
        controls_layout.add_widget(btn_cancel)
        main_layout.add_widget(controls_layout)
        self.add_widget(main_layout)
    
    def load_folders(self):
        self.folders_grid.clear_widgets()
        if not os.path.exists(self.current_path):
            self.folders_grid.add_widget(Label(
                text=reshape_arabic('❌ المجلد غير موجود'),
                font_name=self.app.font_path,
                font_size=40,
                size_hint_y=None,
                height=100,
                color=(1, 0, 0, 1)
            ))
            return
        try:
            items = os.listdir(self.current_path)
            folders = [item for item in items if os.path.isdir(os.path.join(self.current_path, item))]
            if not folders:
                self.folders_grid.add_widget(Label(
                    text=reshape_arabic('📁 لا توجد مجلدات فرعية'),
                    font_name=self.app.font_path,
                    font_size=40,
                    size_hint_y=None,
                    height=100,
                    color=(1, 1, 0, 1)
                ))
                return
            for folder in sorted(folders)[:50]:
                folder_path = os.path.join(self.current_path, folder)
                image_count = 0
                try:
                    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
                    for file in os.listdir(folder_path):
                        if file.lower().endswith(image_extensions):
                            image_count += 1
                except:
                    pass
                folder_container = BoxLayout(
                    orientation='vertical',
                    size_hint_y=None,
                    height=150,
                    padding=5,
                    spacing=2
                )
                with folder_container.canvas.before:
                    Color(0.2, 0.4, 0.8, 0.2)
                    self.folder_rect = Rectangle(pos=folder_container.pos, size=folder_container.size)
                def update_rect(instance, value):
                    instance.canvas.before.clear()
                    with instance.canvas.before:
                        Color(0.2, 0.4, 0.8, 0.2)
                        Rectangle(pos=instance.pos, size=instance.size)
                folder_container.bind(pos=update_rect, size=update_rect)
                folder_icon = Label(
                    text='📁',
                    font_size=60,
                    size_hint_y=0.5,
                    color=(0.2, 0.6, 1, 1)
                )
                folder_name = Label(
                    text=reshape_arabic(folder[:20]),
                    font_name=self.app.font_path,
                    font_size=30,
                    size_hint_y=0.3,
                    color=(1, 1, 1, 1),
                    shorten=True
                )
                folder_info = Label(
                    text=reshape_arabic(f'📸 {image_count} صورة'),
                    font_name=self.app.font_path,
                    font_size=20,
                    size_hint_y=0.2,
                    color=(0, 1, 0, 1) if image_count > 0 else (1, 0, 0, 1)
                )
                select_btn = AnimatedButton(
                    text=reshape_arabic('اختيار'),
                    font_size=25,
                    background_color=(0.2, 0.6, 1, 1),
                    size_hint_y=1
                )
                select_btn.bind(on_press=lambda x, path=folder_path: self.select_this_folder(path))
                folder_container.add_widget(folder_icon)
                folder_container.add_widget(folder_name)
                folder_container.add_widget(folder_info)
                folder_container.add_widget(select_btn)
                self.folders_grid.add_widget(folder_container)
        except Exception as e:
            print(f"خطأ في تحميل المجلدات: {str(e)}")
            self.folders_grid.add_widget(Label(
                text=reshape_arabic(f'❌ خطأ: {str(e)[:30]}'),
                font_name=self.app.font_path,
                font_size=30,
                size_hint_y=None,
                height=100,
                color=(1, 0, 0, 1)
            ))
    
    def select_this_folder(self, folder_path):
        self.selected_folder = folder_path
        self.path_label.text = reshape_arabic(f'📂 {folder_path}')
        for child in self.folders_grid.children:
            if hasattr(child, 'children') and len(child.children) > 0:
                for btn in child.children:
                    if isinstance(btn, AnimatedButton):
                        btn.background_color = (0.2, 0.6, 1, 1)
        for child in self.folders_grid.children:
            if hasattr(child, 'children') and len(child.children) > 0:
                if child.children[0].text == '📁':
                    folder_name = child.children[1].text
                    if folder_name == reshape_arabic(os.path.basename(folder_path)[:20]):
                        for btn in child.children:
                            if isinstance(btn, AnimatedButton):
                                btn.background_color = (0.2, 0.7, 0.3, 1)
                        break
    
    def go_back(self, instance):
        parent_path = os.path.dirname(self.current_path)
        if parent_path and parent_path != self.current_path:
            self.current_path = parent_path
            self.path_label.text = reshape_arabic(f'📂 {self.current_path}')
            self.load_folders()
    
    def go_to_root(self, instance):
        self.current_path = "/storage/emulated/0/صور"
        self.path_label.text = reshape_arabic(f'📂 {self.current_path}')
        self.load_folders()
    
    def select_folder(self, instance):
        if not self.selected_folder:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ يرجى اختيار مجلد أولاً',
                popup_type='warning'
            )
            popup.open()
            return
        self.dismiss()
        gallery = GalleryImport(self.app, self.selected_folder)
        gallery.open()
    
    def cancel(self, instance):
        self.dismiss()
        self.app.freeze_frame = False
        self.app.camera.play = True

class GalleryImport(ModalView):
    def __init__(self, app_instance, folder_path=None, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.95, 0.95)
        self.auto_dismiss = False
        self.background_color = (0, 0, 0, 0.95)
        self.app = app_instance
        self.folder_path = folder_path or "/storage/emulated/0/صور"
        self.selected_images = []
        self.image_widgets = []
        self.build_ui()
    
    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        title = Label(
            text=reshape_arabic('📁 استيراد صور من المجلد'),
            font_name=self.app.font_path,
            font_size=55,
            size_hint_y=0.08,
            bold=True,
            color=(0.2, 0.8, 1, 1)
        )
        main_layout.add_widget(title)
        folder_label = Label(
            text=reshape_arabic(f'📂 المجلد: {self.folder_path}'),
            font_name=self.app.font_path,
            font_size=35,
            size_hint_y=0.05,
            color=(1, 1, 0, 1),
            shorten=True,
            shorten_from='right'
        )
        main_layout.add_widget(folder_label)
        btn_change_folder = AnimatedButton(
            text=reshape_arabic('🔄 تغيير المجلد'),
            font_size=30,
            background_color=(0.5, 0.3, 0.8, 1),
            size_hint_y=0.05
        )
        btn_change_folder.bind(on_press=self.change_folder)
        main_layout.add_widget(btn_change_folder)
        scroll_layout = ScrollView(size_hint_y=0.6)
        self.images_grid = GridLayout(cols=3, spacing=10, padding=10, size_hint_y=None)
        self.images_grid.bind(minimum_height=self.images_grid.setter('height'))
        self.load_images_from_folder()
        scroll_layout.add_widget(self.images_grid)
        main_layout.add_widget(scroll_layout)
        controls_layout = BoxLayout(size_hint_y=0.1, spacing=10, padding=10)
        btn_select_all = AnimatedButton(
            text=reshape_arabic('✅ تحديد الكل'),
            font_size=35,
            background_color=(0.2, 0.6, 1, 1),
            size_hint_x=0.2
        )
        btn_select_all.bind(on_press=self.select_all)
        btn_deselect_all = AnimatedButton(
            text=reshape_arabic('❌ إلغاء الكل'),
            font_size=35,
            background_color=(0.8, 0.2, 0.2, 1),
            size_hint_x=0.2
        )
        btn_deselect_all.bind(on_press=self.deselect_all)
        btn_import = AnimatedButton(
            text=reshape_arabic('📥 استيراد المحدد'),
            font_size=35,
            background_color=(0.2, 0.7, 0.3, 1),
            size_hint_x=0.3
        )
        btn_import.bind(on_press=self.import_selected)
        btn_cancel = AnimatedButton(
            text=reshape_arabic('❌ إلغاء'),
            font_size=35,
            background_color=(0.5, 0.5, 0.5, 1),
            size_hint_x=0.2
        )
        btn_cancel.bind(on_press=self.cancel)
        controls_layout.add_widget(btn_select_all)
        controls_layout.add_widget(btn_deselect_all)
        controls_layout.add_widget(btn_import)
        controls_layout.add_widget(btn_cancel)
        main_layout.add_widget(controls_layout)
        self.add_widget(main_layout)
    
    def change_folder(self, instance):
        self.dismiss()
        folder_selector = FolderSelectionView(self.app)
        folder_selector.open()
    
    def load_images_from_folder(self):
        self.images_grid.clear_widgets()
        self.image_widgets = []
        if not os.path.exists(self.folder_path):
            no_folder_label = Label(
                text=reshape_arabic('❌ المجلد غير موجود'),
                font_name=self.app.font_path,
                font_size=40,
                size_hint_y=None,
                height=100,
                color=(1, 0, 0, 1)
            )
            self.images_grid.add_widget(no_folder_label)
            return
        try:
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
            image_files = []
            for file in os.listdir(self.folder_path):
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(self.folder_path, file))
            if not image_files:
                no_images_label = Label(
                    text=reshape_arabic('❌ لا توجد صور في المجلد'),
                    font_name=self.app.font_path,
                    font_size=40,
                    size_hint_y=None,
                    height=100,
                    color=(1, 1, 0, 1)
                )
                self.images_grid.add_widget(no_images_label)
                return
            for img_path in image_files[:30]:
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        processed_img = preprocess_image_for_recognition(img)
                        rgb_frame = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                        img_container = BoxLayout(
                            orientation='vertical',
                            size_hint_y=None,
                            height=640,
                            padding=5
                        )
                        display_img = img.copy()
                        if face_locations:
                            for (top, right, bottom, left) in face_locations:
                                cv2.rectangle(display_img, (left, top), (right, bottom), (0, 255, 0), 2)
                        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_rgb, (150, 150))
                        img_flip = np.flip(img_resized, 0)
                        texture = Texture.create(size=(150, 150), colorfmt='rgb')
                        texture.blit_buffer(img_flip.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                        img_widget = KivyImage(
                            texture=texture,
                            size_hint=(1, 0.6),
                            allow_stretch=True,
                            keep_ratio=True
                        )
                        filename = os.path.basename(img_path)
                        filename_label = Label(
                            text=reshape_arabic(filename[:15]),
                            font_name=self.app.font_path,
                            font_size=30,
                            size_hint_y=0.1,
                            color=(1, 1, 1, 1),
                            shorten=True
                        )
                        face_status = Label(
                            text=reshape_arabic('✅ يوجد وجه' if face_locations else '❌ لا يوجد وجه'),
                            font_name=self.app.font_path,
                            font_size=30,
                            color=(0, 1, 0, 1) if face_locations else (1, 0, 0, 1),
                            size_hint_y=0.1
                        )
                        select_btn = AnimatedButton(
                            text=reshape_arabic('اختيار'),
                            font_size=22,
                            background_color=(0.5, 0.5, 0.5, 1),
                            size_hint=(1, 0.2)
                        )
                        select_btn.bind(on_press=lambda x, path=img_path, has_face=bool(face_locations): 
                                     self.toggle_image_selection(x, path, has_face))
                        img_container.add_widget(img_widget)
                        img_container.add_widget(filename_label)
                        img_container.add_widget(face_status)
                        img_container.add_widget(select_btn)
                        self.images_grid.add_widget(img_container)
                        self.image_widgets.append({
                            'path': img_path,
                            'container': img_container,
                            'button': select_btn,
                            'face_status': face_status,
                            'has_face': bool(face_locations),
                            'selected': False,
                            'filename': filename
                        })
                except Exception as e:
                    print(f"خطأ في تحميل الصورة {img_path}: {str(e)}")
        except Exception as e:
            print(f"خطأ في قراءة المجلد: {str(e)}")
    
    def toggle_image_selection(self, button, image_path, has_face):
        if not has_face:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ هذه الصورة لا تحتوي على وجه واضح',
                popup_type='warning'
            )
            popup.open()
            return
        for item in self.image_widgets:
            if item['path'] == image_path:
                item['selected'] = not item['selected']
                if item['selected']:
                    item['button'].background_color = (0.2, 0.7, 0.3, 1)
                    item['button'].text = reshape_arabic('✓ تم')
                    self.selected_images.append(image_path)
                else:
                    item['button'].background_color = (0.5, 0.5, 0.5, 1)
                    item['button'].text = reshape_arabic('اختيار')
                    self.selected_images.remove(image_path)
                break
    
    def select_all(self, instance):
        for item in self.image_widgets:
            if item['has_face'] and not item['selected']:
                item['selected'] = True
                item['button'].background_color = (0.2, 0.7, 0.3, 1)
                item['button'].text = reshape_arabic('✓ تم')
                if item['path'] not in self.selected_images:
                    self.selected_images.append(item['path'])
        if not self.selected_images:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ لا توجد صور تحتوي على وجوه',
                popup_type='warning'
            )
            popup.open()
    
    def deselect_all(self, instance):
        for item in self.image_widgets:
            if item['selected']:
                item['selected'] = False
                item['button'].background_color = (0.5, 0.5, 0.5, 1)
                item['button'].text = reshape_arabic('اختيار')
        self.selected_images = []
    
    def import_selected(self, instance):
        if not self.selected_images:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ لم يتم اختيار أي صور',
                popup_type='warning'
            )
            popup.open()
            return
        self.app.multi_face_images = []
        valid_images_count = 0
        processed_folders = set()
        for img_path in self.selected_images:
            img = cv2.imread(img_path)
            if img is not None:
                processed_img = preprocess_image_for_recognition(img)
                rgb_frame = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                if face_locations:
                    self.app.multi_face_images.append(processed_img)
                    valid_images_count += 1
                    folder_name = os.path.basename(os.path.dirname(img_path))
                    processed_folders.add(folder_name)
        if valid_images_count == 0:
            popup = ArabicPopup(
                title='خطأ',
                message='❌ لم يتم العثور على أي وجه في الصور المحددة',
                popup_type='error'
            )
            popup.open()
            return
        self.dismiss()
        if valid_images_count > 0:
            folders_text = "، ".join(processed_folders) if processed_folders else "غير معروف"
            message = f"""
✅ تم استيراد {valid_images_count} صورة بنجاح
📁 المصدر: {folders_text}
📸 سيتم استخدام هذه الصور للتسجيل
"""
            popup = ArabicPopup(
                title='استيراد ناجح',
                message=message,
                popup_type='success'
            )
            popup.bind(on_dismiss=self.go_to_registration)
            popup.open()
    
    def go_to_registration(self, instance):
        self.app.show_registration_form()
    
    def cancel(self, instance):
        self.dismiss()
        self.app.freeze_frame = False
        self.app.camera.play = True

class FaceRecKivyApp(App):
    def build(self):
        Window.fullscreen = True
        Window.clearcolor = (0.95, 0.95, 0.95, 1)
        self.root_layout = BoxLayout(orientation='vertical', spacing=0)
        self.font_path = "/storage/emulated/0/Amiri-Regular.ttf"
        self.last_alert_time = 0
        self.alert_cooldown = 3
        self.person_image_widget = None
        self.confirm_button = None
        self.person_image_data = None
        self.multi_face_images = []
        self.current_image_index = 0
        self.recognition_tolerance = 0.45
        self.min_face_size = 80
        self.use_enhancement = True
        self.images_folder = "/storage/emulated/0/FaceRecognition/Images"
        self.backup_folder = "/storage/emulated/0/FaceRecognition/Backup"
        self.logs_folder = "/storage/emulated/0/FaceRecognition/Logs"
        self.encodings_folder = "/storage/emulated/0/FaceRecognition/Encodings"
        for folder in [self.images_folder, self.backup_folder, self.logs_folder, self.encodings_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        main_pictures_folder = "/storage/emulated/0/صور"
        if not os.path.exists(main_pictures_folder):
            try:
                os.makedirs(main_pictures_folder)
            except:
                pass
        title_bar = BoxLayout(size_hint_y=0.1, padding=[0, 0], spacing=0)
        with title_bar.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.1, 0.3, 0.6,1)
            self.title_rect = Rectangle(pos=title_bar.pos, size=title_bar.size)
        def update_title_rect(instance, value):
            if hasattr(self, 'title_rect'):
                self.title_rect.pos = instance.pos
                self.title_rect.size = instance.size
        title_bar.bind(pos=update_title_rect, size=update_title_rect)
        try:
            self.left_gif = KivyImage(
                source='/storage/emulated/0/lazreg.gif',
                size_hint_x=0.15,
                allow_stretch=True,
                keep_ratio=True,
                anim_delay=0.1,
                anim_loop=0
            )
        except:
            self.left_gif = Label(
                text='👤',
                font_size=70,
                size_hint_x=0.15,
                color=(1, 1, 1, 1)
            )
        title_label = Label(
            text=reshape_arabic('نظام التعرف على بصمة الوجه'),
            font_name=self.font_path,
            font_size=55,
            bold=True,
            color=(1, 1, 1, 1),
            halign='center',
            valign='middle',
            size_hint_x=0.7
        )
        title_label.bind(size=title_label.setter('text_size'))
        try:
            self.right_gif = KivyImage(
                source='/storage/emulated/0/lazreg.gif',
                size_hint_x=0.15,
                allow_stretch=True,
                keep_ratio=True,
                anim_delay=0.1,
                anim_loop=0
            )
        except:
            self.right_gif = Label(
                text='📸',
                font_size=70,
                size_hint_x=0.15,
                color=(1, 1, 1, 1)
            )
        title_bar.add_widget(self.left_gif)
        title_bar.add_widget(title_label)
        title_bar.add_widget(self.right_gif)
        self.root_layout.add_widget(title_bar)
        self.data_layout = BoxLayout(orientation='vertical', size_hint_y=0.25, padding=[15, 10], spacing=5)
        with self.data_layout.canvas.before:
            Color(0.95, 0.95, 0.98, 1)
            self.data_rect = Rectangle(pos=self.data_layout.pos, size=self.data_layout.size)
        def update_data_rect(instance, value):
            if hasattr(self, 'data_rect'):
                self.data_rect.pos = instance.pos
                self.data_rect.size = instance.size
        self.data_layout.bind(pos=update_data_rect, size=update_data_rect)
        data_title = Label(
            text=reshape_arabic('📋 بيانات الشخص'),
            font_name=self.font_path,
            font_size=45,
            size_hint_y=0.1,
            halign='center',
            valign='middle',
            color=(0.1, 0.3, 0.6, 1),
            bold=True
        )
        data_title.bind(size=data_title.setter('text_size'))
        self.data_layout.add_widget(data_title)
        self.person_info_layout = BoxLayout(orientation='horizontal', size_hint_y=0.8, spacing=10)
        self.image_container = BoxLayout(orientation='vertical', size_hint_x=0.4, padding=[5, 5])
        self.person_image_widget = KivyImage(
            size_hint=(1, 0.9),
            allow_stretch=True,
            keep_ratio=True,
            opacity=0
        )
        self.confirm_button = AnimatedButton(
            text=reshape_arabic('✅ تأكيد'),
            font_size=35,
            background_color=(0.2, 0.7, 0.3, 1),
            size_hint=(1, 0.1),
            opacity=0,
            disabled=True
        )
        self.confirm_button.bind(on_press=self.clear_person_data)
        self.image_container.add_widget(self.person_image_widget)
        self.image_container.add_widget(self.confirm_button)
        scroll_data = ScrollView(size_hint_x=0.6)
        self.person_data_container = GridLayout(cols=1, size_hint_y=None, spacing=5, padding=[10, 5])
        self.person_data_container.bind(minimum_height=self.person_data_container.setter('height'))
        self.person_info_labels = []
        info_fields = ['👤 الاسم:', '💼 المهنة:', '🆔 رقم التعريف:', '📊 مرات التعرف:', '📅 آخر ظهور:', '📞 الهاتف:', '📍 العنوان:', '📸 عدد الصور:']
        for field in info_fields:
            label = Label(
                text=reshape_arabic(field + ' --'),
                font_name=self.font_path,
                font_size=38,
                size_hint_y=None,
                height=50,
                halign='right',
                valign='middle',
                color=(0, 0, 0, 1),
                bold=True if field == '👤 الاسم:' else False
            )
            label.bind(size=label.setter('text_size'))
            self.person_info_labels.append(label)
            self.person_data_container.add_widget(label)
        scroll_data.add_widget(self.person_data_container)
        self.person_info_layout.add_widget(self.image_container)
        self.person_info_layout.add_widget(scroll_data)
        self.data_layout.add_widget(self.person_info_layout)
        self.root_layout.add_widget(self.data_layout)
        from kivy.uix.camera import Camera
        self.camera = Camera(play=True, resolution=(640, 480))
        self.camera.size_hint_y = 0.5
        self.camera.allow_stretch = True
        self.camera.keep_ratio = False
        self.root_layout.add_widget(self.camera)
        try:
            self.font = ImageFont.truetype(self.font_path, 24)
            self.font_big = ImageFont.truetype(self.font_path, 32)
        except:
            self.font = ImageFont.load_default()
            self.font_big = ImageFont.load_default()
        self.freeze_frame = False
        self.captured_frame = None
        self.current_detected_person = None
        self.face_locations = None
        self.detected_registered_person = None
        self.detected_unregistered_face = False
        control_layout = BoxLayout(orientation='horizontal', size_hint_y=0.08, spacing=5, padding=[10, 0])
        with control_layout.canvas.before:
            Color(0.2, 0.2, 0.2, 0.5)
            self.control_rect = Rectangle(pos=control_layout.pos, size=control_layout.size)
        def update_control_rect(instance, value):
            if hasattr(self, 'control_rect'):
                self.control_rect.pos = instance.pos
                self.control_rect.size = instance.size
        control_layout.bind(pos=update_control_rect, size=update_control_rect)
        self.btn_cancel = AnimatedButton(
            text=reshape_arabic('❌ إلغاء'),
            font_size=30,
            background_color=(0.8, 0.2, 0.2, 1),
            size_hint_x=0.15
        )
        self.btn_register = AnimatedButton(
            text=reshape_arabic('📝 تسجيل جديد'),
            font_size=30,
            background_color=(0.2, 0.7, 0.3, 1),
            size_hint_x=0.2
        )
        self.btn_gallery = AnimatedButton(
            text=reshape_arabic('🖼️ استيراد صور'),
            font_size=30,
            background_color=(0.5, 0.3, 0.8, 1),
            size_hint_x=0.2
        )
        self.btn_gallery.bind(on_press=self.show_folder_selector)
        self.btn_settings = AnimatedButton(
            text=reshape_arabic('⚙️ إعدادات'),
            font_size=30,
            background_color=(0.5, 0.5, 0.5, 1),
            size_hint_x=0.15
        )
        self.btn_exit = AnimatedButton(
            text=reshape_arabic('🚪 خروج'),
            font_size=30,
            background_color=(0.6, 0.2, 0.6, 1),
            size_hint_x=0.15
        )
        self.btn_exit.bind(on_press=self.exit_app)
        self.btn_cancel.bind(on_press=self.on_cancel)
        self.btn_register.bind(on_press=self.show_input_form)
        self.btn_settings.bind(on_press=self.show_settings)
        control_layout.add_widget(self.btn_cancel)
        control_layout.add_widget(self.btn_register)
        control_layout.add_widget(self.btn_gallery)
        control_layout.add_widget(self.btn_settings)
        control_layout.add_widget(self.btn_exit)
        self.root_layout.add_widget(control_layout)
        self.sound_enabled = True
        try:
            self.sound_success = SoundLoader.load('/storage/emulated/0/salam.wav')
            self.sound_error = SoundLoader.load('/storage/emulated/0/inconu.wav')
            self.sound_alert = self.sound_error
        except:
            self.sound_success = SoundLoader.load('/storage/emulated/0/inconu.wav')
            self.sound_detection = self.sound_success
            self.sound_error = self.sound_success
            self.sound_alert = self.sound_error
        self.registered_faces_cache = None
        self.init_database()
        Clock.schedule_interval(self.update_registered_faces_cache, 5)
        Clock.schedule_interval(self.update, 1/30)
        return self.root_layout
    
    def show_folder_selector(self, instance):
        self.freeze_frame = True
        folder_selector = FolderSelectionView(self)
        folder_selector.open()
    
    def exit_app(self, instance):
        popup = ModalView(size_hint=(0.85, 0.5), auto_dismiss=False)
        popup.background_color = (0, 0, 0, 0.9)
        layout = BoxLayout(orientation='vertical', padding=25, spacing=20)
        title_label = Label(
            text=reshape_arabic('تأكيد الخروج'),
            font_name=self.font_path,
            font_size=55,
            size_hint=(1, 0.25),
            halign='center',
            valign='middle',
            color=(1, 0.6, 0, 1),
            bold=True
        )
        title_label.bind(size=title_label.setter('text_size'))
        message_label = Label(
            text=reshape_arabic('هل تريد إغلاق التطبيق؟'),
            font_name=self.font_path,
            font_size=45,
            size_hint=(1, 0.5),
            halign='center',
            valign='middle',
            color=(1, 1, 1, 1)
        )
        message_label.bind(size=message_label.setter('text_size'))
        buttons_bar = BoxLayout(size_hint=(1, 0.2), spacing=15)
        btn_confirm = AnimatedButton(
            text=reshape_arabic('✅ نعم'),
            font_size=50,
            background_color=(0.2, 0.7, 0.3, 1),
            size_hint_x=0.5
        )
        btn_confirm.bind(on_press=lambda x: self.stop())
        btn_cancel = AnimatedButton(
            text=reshape_arabic('❌ لا'),
            font_size=50,
            background_color=(0.8, 0.2, 0.2, 1),
            size_hint_x=0.5
        )
        btn_cancel.bind(on_press=popup.dismiss)
        buttons_bar.add_widget(btn_confirm)
        buttons_bar.add_widget(btn_cancel)
        layout.add_widget(title_label)
        layout.add_widget(message_label)
        layout.add_widget(buttons_bar)
        popup.add_widget(layout)
        popup.open()
    
    def init_database(self):
        self.db_path = "/storage/emulated/0/FaceRecognition/database.db"
        db_folder = os.path.dirname(self.db_path)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lastname TEXT NOT NULL,
                firstname TEXT NOT NULL,
                job TEXT NOT NULL,
                phone TEXT,
                address TEXT,
                images_count INTEGER DEFAULT 1,
                encodings_path TEXT NOT NULL,
                main_image_path TEXT NOT NULL,
                registration_date TEXT NOT NULL,
                last_seen TEXT,
                times_recognized INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                person_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        conn.commit()
        conn.close()
        self.update_registered_faces_cache()
    
    def play_sound(self, sound_type='detection'):
        if self.sound_enabled:
            sound = getattr(self, f'sound_{sound_type}', None)
            if sound:
                try:
                    sound.play()
                except:
                    pass
    
    def log_recognition(self, person_id, person_name, confidence=0.9):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('UPDATE persons SET last_seen = ?, times_recognized = times_recognized + 1 WHERE id = ?', (timestamp, person_id))
            cursor.execute('INSERT INTO attendance_log (person_id, person_name, timestamp, confidence) VALUES (?, ?, ?, ?)', (person_id, person_name, timestamp, confidence))
            conn.commit()
            conn.close()
            self.save_log_to_file(person_name, timestamp)
        except Exception as e:
            print(f"خطأ في تسجيل الحضور: {str(e)}")
    
    def save_log_to_file(self, person_name, timestamp):
        try:
            date = datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(self.logs_folder, f"attendance_{date}.txt")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} - تم التعرف على: {person_name}\n")
        except Exception as e:
            print(f"خطأ في حفظ ملف السجل: {str(e)}")
    
    def update_registered_faces_cache(self, dt=None):
        self.registered_faces_cache = self.get_all_registered_faces()
        print(f"تم تحديث الذاكرة المؤقتة: {len(self.registered_faces_cache)} شخص مسجل")
    
    def get_all_registered_faces(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, firstname, lastname, job, phone, address, encodings_path, main_image_path, images_count, times_recognized, last_seen FROM persons')
            persons = cursor.fetchall()
            conn.close()
            registered_faces = []
            for person in persons:
                try:
                    encodings_path = person[6]
                    if os.path.exists(encodings_path):
                        with open(encodings_path, 'rb') as f:
                            face_encodings = pickle.load(f)
                        registered_faces.append({
                            'id': person[0],
                            'firstname': person[1],
                            'lastname': person[2],
                            'job': person[3],
                            'phone': person[4],
                            'address': person[5],
                            'encodings': face_encodings,
                            'main_image_path': person[7],
                            'images_count': person[8],
                            'times_recognized': person[9],
                            'last_seen': person[10]
                        })
                        print(f"تم تحميل بصمات الشخص: {person[1]} {person[2]}")
                except Exception as e:
                    print(f"خطأ في تحميل بصمات الشخص {person[1]}: {str(e)}")
            return registered_faces
        except Exception as e:
            print(f"خطأ في جلب البيانات من قاعدة البيانات: {str(e)}")
            return []
    
    def check_if_face_registered(self, face_encoding):
        if not self.registered_faces_cache or face_encoding is None:
            return None, 0
        best_match = None
        best_distance = float('inf')
        for registered in self.registered_faces_cache:
            try:
                for encoding in registered['encodings']:
                    distance = face_recognition.face_distance([encoding], face_encoding)[0]
                    if distance < best_distance:
                        best_distance = distance
                    if distance < self.recognition_tolerance:
                        print(f"تم العثور على تطابق: {registered['firstname']} {registered['lastname']} بمسافة {distance}")
                        return registered, distance
            except Exception as e:
                print(f"خطأ في المقارنة: {str(e)}")
                pass
        return None, best_distance
    
    def display_person_image(self, person):
        try:
            if person and 'main_image_path' in person and os.path.exists(person['main_image_path']):
                self.person_image_widget.source = person['main_image_path']
                self.person_image_widget.opacity = 1
                self.person_image_widget.reload()
                print(f"تم تحميل الصورة: {person['main_image_path']}")
            else:
                print("لم يتم العثور على صورة للشخص")
                self.person_image_widget.opacity = 0
        except Exception as e:
            print(f"خطأ في عرض الصورة: {str(e)}")
            self.person_image_widget.opacity = 0
    
    def update_person_data_display(self, person=None):
        if person:
            data_values = [
                f"👤 الاسم: {person['firstname']} {person['lastname']}",
                f"💼 المهنة: {person['job']}",
                f"🆔 رقم التعريف: {person['id']}",
                f"📊 مرات التعرف: {person.get('times_recognized', 0)}",
                f"📅 آخر ظهور: {person.get('last_seen', 'لم يسبق')}",
                f"📞 الهاتف: {person.get('phone', 'غير محدد')}",
                f"📍 العنوان: {person.get('address', 'غير محدد')}",
                f"📸 عدد الصور: {person.get('images_count', 1)}"
            ]
            for i, label in enumerate(self.person_info_labels):
                if i < len(data_values):
                    label.text = reshape_arabic(data_values[i])
            self.display_person_image(person)
            self.confirm_button.opacity = 1
            self.confirm_button.disabled = False
            self.person_image_data = person
        else:
            empty_values = [
                '👤 الاسم: --',
                '💼 المهنة: --',
                '🆔 رقم التعريف: --',
                '📊 مرات التعرف: --',
                '📅 آخر ظهور: --',
                '📞 الهاتف: --',
                '📍 العنوان: --',
                '📸 عدد الصور: --'
            ]
            for i, label in enumerate(self.person_info_labels):
                label.text = reshape_arabic(empty_values[i])
            self.person_image_widget.opacity = 0
            self.confirm_button.opacity = 0
            self.confirm_button.disabled = True
            self.person_image_data = None
    
    def clear_person_data(self, instance):
        self.update_person_data_display(None)
        self.freeze_frame = False
        self.captured_frame = None
        self.detected_registered_person = None
        self.current_detected_person = None
        self.detected_unregistered_face = False
        self.camera.play = True
    
    def draw_arabic_on_frame(self, frame, text, position, color=(0,255,0), font_size=32):
        try:
            img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(self.font_path, font_size)
            reshaped_text = arabic_reshaper.reshape(text)
            display_text = reshaped_text[::-1]
            draw.text(position, display_text, font=font, fill=color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return frame
    
    def update(self, dt):
        if self.freeze_frame:
            return
        if not self.camera.texture:
            return
        try:
            texture = self.camera.texture
            size = texture.size
            pixels = texture.pixels
            frame = np.frombuffer(pixels, dtype=np.uint8)
            frame = frame.reshape(size[1], size[0], 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.flip(frame, 1)
            if self.use_enhancement:
                enhanced_frame = enhance_face_for_recognition(frame)
                rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            display_frame = frame.copy()
            if self.face_locations and not self.freeze_frame:
                try:
                    current_face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)
                    self.detected_unregistered_face = False
                    for (top, right, bottom, left), face_encoding in zip(self.face_locations, current_face_encodings):
                        registered_person, distance = self.check_if_face_registered(face_encoding)
                        if registered_person:
                            self.detected_registered_person = registered_person
                            self.current_detected_person = registered_person
                            self.freeze_frame = True
                            self.captured_frame = frame.copy()
                            self.play_sound('success')
                            person_name = f"{registered_person['firstname']} {registered_person['lastname']}"
                            similarity = max(0, min(100, (1 - distance) * 100))
                            self.log_recognition(registered_person['id'], person_name, similarity/100)
                            self.update_person_data_display(registered_person)
                            message = f"""
👤 الاسم: {registered_person['firstname']} {registered_person['lastname']}
💼 المهنة: {registered_person['job']}
🆔 رقم التسجيل: {registered_person['id']}
📊 عدد مرات التعرف: {registered_person.get('times_recognized', 0) + 1}
📸 عدد الصور المسجلة: {registered_person.get('images_count', 1)}
🔍 نسبة التطابق: {similarity:.1f}%
"""
                            popup = ArabicPopup(title='✅ تم التعرف بنجاح', message=message, popup_type='success')
                            popup.bind(on_dismiss=self.on_popup_dismiss)
                            popup.open()
                            break
                        else:
                            self.detected_unregistered_face = True
                            self.captured_frame = frame.copy()
                            self.detected_registered_person = None
                            self.current_detected_person = None
                            current_time = time.time()
                            if current_time - self.last_alert_time > self.alert_cooldown:
                                self.play_sound('alert')
                                self.last_alert_time = current_time
                            cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 0, 0), 4)
                            similarity = max(0, (1 - distance) * 100) if distance != float('inf') else 0
                            display_frame = self.draw_arabic_on_frame(
                                display_frame, f"❌ غير مسجل ({similarity:.1f}%)", (left, top - 30), (255, 0, 0), 24
                            )
                            self.update_person_data_display(None)
                except Exception as e:
                    print(f"خطأ في عملية التعرف: {str(e)}")
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            frame_flip = np.flip(frame_rgb, 0)
            texture = Texture.create(size=(frame_flip.shape[1], frame_flip.shape[0]), colorfmt='rgba')
            texture.blit_buffer(frame_flip.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
            self.camera.texture = texture
        except Exception as e:
            print(f"خطأ عام في التحديث: {str(e)}")
    
    def on_popup_dismiss(self, instance):
        pass
    
    def add_face_image(self):
        if self.captured_frame is not None:
            processed_img = preprocess_image_for_recognition(self.captured_frame)
            self.multi_face_images.append(processed_img)
            if len(self.multi_face_images) < 5:
                message = f"""
✅ تم إضافة صورة {len(self.multi_face_images)}
📸 التقط المزيد من الزوايا المختلفة
"""
                popup = ArabicPopup(
                    title='صورة مضافة',
                    message=message,
                    popup_type='success'
                )
                popup.bind(on_dismiss=self.continue_capture)
                popup.open()
            else:
                self.show_registration_form()
    
    def continue_capture(self, instance):
        enhancer = ImageEnhancer(self.captured_frame, self)
        enhancer.open()
    
    def show_input_form(self, instance):
        if self.detected_registered_person is not None:
            message = f"""
⚠️ هذا الشخص مسجل بالفعل!

👤 {self.detected_registered_person['firstname']} {self.detected_registered_person['lastname']}
💼 {self.detected_registered_person['job']}
🆔 رقم: {self.detected_registered_person['id']}
📸 عدد الصور المسجلة: {self.detected_registered_person.get('images_count', 1)}
"""
            popup = ArabicPopup(title='تنبيه', message=message, popup_type='warning')
            popup.open()
            def reset_after_popup(dt):
                self.freeze_frame = False
                self.captured_frame = None
                self.detected_registered_person = None
                self.current_detected_person = None
                self.detected_unregistered_face = False
                self.camera.play = True
                self.update_person_data_display(None)
            Clock.schedule_once(reset_after_popup, 3)
            return
        if self.detected_unregistered_face:
            self.freeze_frame = True
            self.multi_face_images = []
            capture = MultiFaceCapture(self)
            capture.open()
        else:
            popup = ArabicPopup(
                title='تنبيه',
                message='❌ لم يتم اكتشاف وجه!\nيرجى التقاط صورة للوجه أولاً',
                popup_type='error'
            )
            popup.open()
    
    def show_registration_form(self):
        self.root_layout.remove_widget(self.camera)
        self.data_layout.size_hint_y = 0
        self.data_layout.opacity = 0
        self.data_layout.disabled = True
        self.input_layout = BoxLayout(orientation='vertical', spacing=0, padding=[0, 0], size_hint_y=0.6)
        images_count_label = Label(
            text=reshape_arabic(f'📸 تم اختيار {len(self.multi_face_images)} صورة'),
            font_name=self.font_path,
            font_size=40,
            size_hint_y=0.1,
            color=(0.2, 0.8, 0.2, 1)
        )
        self.input_layout.add_widget(images_count_label)
        scroll_view = ScrollView(size_hint_y=2)
        fields_container = GridLayout(cols=1, size_hint_y=None, spacing=5, padding=[10, 5])
        fields_container.bind(minimum_height=fields_container.setter('height'))
        self.lastname_field = ArabicTextInput('👤 اللقب', 'أدخل اللقب')
        self.firstname_field = ArabicTextInput('👤 الاسم', 'أدخل الاسم')
        self.job_field = ArabicTextInput('💼 المهنة', 'أدخل المهنة')
        self.phone_field = ArabicTextInput('📞 رقم الهاتف', 'أدخل رقم الهاتف (اختياري)')
        self.address_field = ArabicTextInput('📍 العنوان', 'أدخل العنوان (اختياري)')
        fields_container.add_widget(self.lastname_field)
        fields_container.add_widget(self.firstname_field)
        fields_container.add_widget(self.job_field)
        fields_container.add_widget(self.phone_field)
        fields_container.add_widget(self.address_field)
        fields_container.height = len(fields_container.children) * 160
        scroll_view.add_widget(fields_container)
        self.input_layout.add_widget(scroll_view)
        buttons_bar = BoxLayout(size_hint_y=0.5, spacing=5, padding=[10, 5])
        btn_save = AnimatedButton(
            text=reshape_arabic('💾 حفظ البيانات'),
            font_size=38,
            background_color=(0.2, 0.7, 0.3, 1)
        )
        btn_save.bind(on_press=self.save_data)
        btn_cancel_form = AnimatedButton(
            text=reshape_arabic('❌ إلغاء'),
            font_size=38,
            background_color=(0.8, 0.2, 0.2, 1)
        )
        btn_cancel_form.bind(on_press=self.cancel_registration)
        buttons_bar.add_widget(btn_save)
        buttons_bar.add_widget(btn_cancel_form)
        self.input_layout.add_widget(buttons_bar)
        self.root_layout.add_widget(self.input_layout, index=1)
        self.empty_space = BoxLayout(size_hint_y=0.3)
        with self.empty_space.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.95, 0.95, 0.95, 1)
            self.empty_rect = Rectangle(pos=self.empty_space.pos, size=self.empty_space.size)
        def update_empty_rect(instance, value):
            if hasattr(self, 'empty_rect'):
                self.empty_rect.pos = instance.pos
                self.empty_rect.size = instance.size
        self.empty_space.bind(pos=update_empty_rect, size=update_empty_rect)
        self.root_layout.add_widget(self.empty_space)
    
    def cancel_registration(self, instance):
        if hasattr(self, 'input_layout') and self.input_layout:
            self.root_layout.remove_widget(self.input_layout)
            self.input_layout = None
        if hasattr(self, 'empty_space') and self.empty_space:
            self.root_layout.remove_widget(self.empty_space)
            self.empty_space = None
        self.root_layout.add_widget(self.camera, index=1)
        self.data_layout.size_hint_y = 0.35
        self.data_layout.opacity = 1
        self.data_layout.disabled = False
        self.freeze_frame = False
        self.captured_frame = None
        self.detected_registered_person = None
        self.current_detected_person = None
        self.detected_unregistered_face = False
        self.camera.play = True
        self.multi_face_images = []
    
    def save_images_with_faces(self, firstname, lastname):
        if not self.multi_face_images:
            return None, 0, None
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            main_image_path = None
            for i, img in enumerate(self.multi_face_images):
                filename = f"{firstname}_{lastname}_{timestamp}_{i+1}.jpg"
                image_path = os.path.join(self.images_folder, filename)
                cv2.imwrite(image_path, img)
                cv2.imwrite(os.path.join(self.backup_folder, filename), img)
                if i == 0:
                    main_image_path = image_path
            face_encodings = []
            for img in self.multi_face_images:
                processed_img = preprocess_image_for_recognition(img)
                rgb_frame = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                if face_locations:
                    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if encodings:
                        face_encodings.extend(encodings)
            if not face_encodings:
                print("تحذير: لم يتم استخراج أي بصمات وجه من الصور")
                return None, 0, None
            encodings_filename = f"{firstname}_{lastname}_{timestamp}.pkl"
            encodings_path = os.path.join(self.encodings_folder, encodings_filename)
            with open(encodings_path, 'wb') as f:
                pickle.dump(face_encodings, f)
            print(f"تم حفظ {len(face_encodings)} بصمة وجه في {encodings_path}")
            return encodings_path, len(face_encodings), main_image_path
        except Exception as e:
            print(f"خطأ في حفظ الصور: {str(e)}")
            return None, 0, None
    
    def save_to_database(self, data):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            registration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO persons (
                    lastname, firstname, job, phone, address,
                    encodings_path, main_image_path, images_count, registration_date, last_seen, times_recognized
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['lastname'], data['firstname'], data['job'],
                data['phone'], data['address'], data['encodings_path'],
                data['main_image_path'], data['images_count'], registration_date, registration_date, 1
            ))
            person_id = cursor.lastrowid
            conn.commit()
            conn.close()
            print(f"تم حفظ الشخص في قاعدة البيانات بالرقم {person_id}")
            self.update_registered_faces_cache()
            return person_id
        except Exception as e:
            print(f"خطأ في حفظ البيانات: {str(e)}")
            return None
    
    def save_data(self, instance):
        lastname = self.lastname_field.get_text().strip()
        firstname = self.firstname_field.get_text().strip()
        job = self.job_field.get_text().strip()
        if not lastname or not firstname or not job:
            popup = ArabicPopup(title='تنبيه', message='❌ يرجى ملء جميع الحقول المطلوبة', popup_type='warning')
            popup.open()
            return
        if not self.multi_face_images:
            popup = ArabicPopup(title='تنبيه', message='❌ لا توجد صور للوجه', popup_type='error')
            popup.open()
            return
        all_encodings = []
        valid_images = []
        for img in self.multi_face_images:
            processed_img = preprocess_image_for_recognition(img)
            rgb_frame = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            if face_locations:
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if encodings:
                    all_encodings.extend(encodings)
                    valid_images.append(processed_img)
        if not all_encodings:
            popup = ArabicPopup(title='تنبيه', message='❌ لم يتم استخراج بصمات الوجه من الصور', popup_type='error')
            popup.open()
            return
        self.multi_face_images = valid_images
        for encoding in all_encodings:
            registered_person, distance = self.check_if_face_registered(encoding)
            if registered_person:
                similarity = (1 - distance) * 100
                message = f"""
⚠️ هذا الوجه مسجل مسبقاً!
الاسم: {registered_person['firstname']} {registered_person['lastname']}
المهنة: {registered_person['job']}
رقم التسجيل: {registered_person['id']}
نسبة التطابق: {similarity:.1f}%
"""
                popup = ArabicPopup(title='تنبيه', message=message, popup_type='warning')
                popup.open()
                return
        data = {
            'lastname': lastname,
            'firstname': firstname,
            'job': job,
            'phone': self.phone_field.get_text().strip() or 'غير محدد',
            'address': self.address_field.get_text().strip() or 'غير محدد'
        }
        encodings_path, images_count, main_image_path = self.save_images_with_faces(data['firstname'], data['lastname'])
        if not encodings_path:
            popup = ArabicPopup(title='خطأ', message='❌ تعذر حفظ الصور', popup_type='error')
            popup.open()
            return
        data['encodings_path'] = encodings_path
        data['images_count'] = images_count
        data['main_image_path'] = main_image_path
        person_id = self.save_to_database(data)
        if person_id:
            message = f"""
✅ تم التسجيل بنجاح!
🆔 رقم التسجيل: {person_id}
👤 الاسم: {data['firstname']} {data['lastname']}
💼 المهنة: {data['job']}
📞 الهاتف: {data['phone']}
📸 عدد الصور: {images_count}
🎯 عتبة التعريف: {self.recognition_tolerance}
تم حفظ بصمات الوجه بنجاح
"""
            popup = ArabicPopup(title='🎉 تهانينا!', message=message, popup_type='success')
            popup.bind(on_dismiss=self.on_registration_complete)
            popup.open()
        else:
            popup = ArabicPopup(title='خطأ', message='❌ تعذر حفظ البيانات\nيرجى المحاولة مرة أخرى', popup_type='error')
            popup.open()
    
    def on_registration_complete(self, instance):
        self.cancel_registration(None)
    
    def toggle_setting(self, setting):
        if setting == 'sound':
            self.sound_enabled = not self.sound_enabled
        elif setting == 'tolerance':
            tolerances = [0.35, 0.4, 0.45, 0.5]
            current_idx = tolerances.index(self.recognition_tolerance) if self.recognition_tolerance in tolerances else 2
            next_idx = (current_idx + 1) % len(tolerances)
            self.recognition_tolerance = tolerances[next_idx]
        elif setting == 'enhancement':
            self.use_enhancement = not self.use_enhancement
        self.show_settings(None)
    
    def show_settings(self, instance):
        self.freeze_frame = True
        if hasattr(self, 'settings_popup') and self.settings_popup:
            self.settings_popup.dismiss()
        self.settings_popup = ModalView(size_hint=(0.9, 0.8), auto_dismiss=False)
        self.settings_popup.background_color = (0, 0, 0, 0.9)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        image_layout = BoxLayout(size_hint_y=0.2, padding=[10, 5])
        try:
            settings_image = KivyImage(
                source='/storage/emulated/0/img.jpg',
                allow_stretch=True,
                keep_ratio=True,
                size_hint=(1, 1)
            )
        except:
            settings_image = Label(
                text='⚙️ الإعدادات',
                font_name=self.font_path,
                font_size=45,
                color=(1, 1, 1, 1),
                size_hint=(1, 1)
            )
        image_layout.add_widget(settings_image)
        layout.add_widget(image_layout)
        title = Label(
            text=reshape_arabic('⚙️ لوحة التحكم والإعدادات ⚙️'),
            font_name=self.font_path,
            font_size=55,
            bold=True,
            color=(0.2, 0.6, 1, 1),
            halign='center',
            size_hint_y=0.1
        )
        title.bind(size=title.setter('text_size'))
        layout.add_widget(title)
        sound_layout = BoxLayout(size_hint_y=0.1, spacing=20)
        sound_label = Label(
            text=reshape_arabic('🔊 تشغيل الأصوات:'),
            font_name=self.font_path,
            font_size=45,
            halign='right',
            valign='middle',
            size_hint_x=0.5
        )
        sound_label.bind(size=sound_label.setter('text_size'))
        sound_btn = AnimatedButton(
            text=reshape_arabic('✅ نعم' if self.sound_enabled else '❌ لا'),
            font_size=45,
            background_color=(0.2, 0.7, 0.3, 1) if self.sound_enabled else (0.8, 0.2, 0.2, 1),
            size_hint_x=0.5
        )
        sound_btn.bind(on_press=lambda x: self.toggle_setting('sound'))
        sound_layout.add_widget(sound_label)
        sound_layout.add_widget(sound_btn)
        layout.add_widget(sound_layout)
        tolerance_layout = BoxLayout(size_hint_y=0.1, spacing=20)
        tolerance_label = Label(
            text=reshape_arabic('📊 عتبة التعريف:'),
            font_name=self.font_path,
            font_size=45,
            halign='right',
            valign='middle',
            size_hint_x=0.5
        )
        tolerance_label.bind(size=tolerance_label.setter('text_size'))
        tolerance_btn = AnimatedButton(
            text=reshape_arabic(f'{self.recognition_tolerance}'),
            font_size=45,
            background_color=(0.2, 0.6, 1, 1),
            size_hint_x=0.5
        )
        tolerance_btn.bind(on_press=lambda x: self.toggle_setting('tolerance'))
        tolerance_layout.add_widget(tolerance_label)
        tolerance_layout.add_widget(tolerance_btn)
        layout.add_widget(tolerance_layout)
        enhancement_layout = BoxLayout(size_hint_y=0.1, spacing=20)
        enhancement_label = Label(
            text=reshape_arabic('👓 تحسين للنظارات:'),
            font_name=self.font_path,
            font_size=45,
            halign='right',
            valign='middle',
            size_hint_x=0.5
        )
        enhancement_label.bind(size=enhancement_label.setter('text_size'))
        enhancement_btn = AnimatedButton(
            text=reshape_arabic('✅ مفعل' if self.use_enhancement else '❌ معطل'),
            font_size=45,
            background_color=(0.2, 0.7, 0.3, 1) if self.use_enhancement else (0.8, 0.2, 0.2, 1),
            size_hint_x=0.5
        )
        enhancement_btn.bind(on_press=lambda x: self.toggle_setting('enhancement'))
        enhancement_layout.add_widget(enhancement_label)
        enhancement_layout.add_widget(enhancement_btn)
        layout.add_widget(enhancement_layout)
        info_layout = BoxLayout(orientation='vertical', size_hint_y=0.3, spacing=10)
        info_label = Label(
            text=reshape_arabic('ℹ️ معلومات النظام'),
            font_name=self.font_path,
            font_size=50,
            bold=True,
            color=(1, 1, 1, 1),
            halign='center',
            size_hint_y=0.2
        )
        info_label.bind(size=info_label.setter('text_size'))
        info_layout.add_widget(info_label)
        persons_count = len(self.registered_faces_cache) if self.registered_faces_cache else 0
        info_text = f"""
📊 عدد المسجلين: {persons_count}
💾 حجم الصور: {self.get_folder_size(self.images_folder)}
🎯 عتبة التعريف: {self.recognition_tolerance} (أقل = تسامح أكبر)
👓 تحسين النظارات: {'✅' if self.use_enhancement else '❌'}
📸 تسجيل متعدد الزوايا: ✅ مفعل
📱 الإصدار: 5.2 

لزرق عبدالقادر الاغواط ....الجزائر
تصميم وبرمجة
"""
        info_content = Label(
            text=reshape_arabic(info_text),
            font_name=self.font_path,
            font_size=40,
            halign='right',
            valign='top',
            color=(0.8, 0.8, 0.8, 1),
            size_hint_y=0.8
        )
        info_content.bind(size=info_content.setter('text_size'))
        info_layout.add_widget(info_content)
        layout.add_widget(info_layout)
        btn_close = AnimatedButton(
            text=reshape_arabic('🔙 العودة'),
            font_size=48,
            background_color=(0.2, 0.6, 1, 1),
            size_hint_y=0.1
        )
        btn_close.bind(on_press=lambda x: self.close_settings(self.settings_popup))
        layout.add_widget(btn_close)
        self.settings_popup.add_widget(layout)
        self.settings_popup.open()
    
    def close_settings(self, popup=None):
        if popup:
            popup.dismiss()
        elif hasattr(self, 'settings_popup') and self.settings_popup:
            self.settings_popup.dismiss()
        self.freeze_frame = False
        self.camera.play = True
    
    def get_folder_size(self, folder_path):
        total_size = 0
        try:
            for path, dirs, files in os.walk(folder_path):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)
            if total_size < 1024:
                return f"{total_size} B"
            elif total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            else:
                return f"{total_size / (1024 * 1024):.1f} MB"
        except:
            return "0 B"
    
    def on_cancel(self, instance):
        self.freeze_frame = False
        self.captured_frame = None
        self.detected_registered_person = None
        self.current_detected_person = None
        self.detected_unregistered_face = False
        self.camera.play = True
        self.update_person_data_display(None)
        if hasattr(self, 'input_layout') and self.input_layout:
            self.root_layout.remove_widget(self.input_layout)
            self.input_layout = None
        if hasattr(self, 'empty_space') and self.empty_space:
            self.root_layout.remove_widget(self.empty_space)
            self.empty_space = None
        if self.camera not in self.root_layout.children:
            self.root_layout.add_widget(self.camera, index=1)
        self.data_layout.size_hint_y = 0.35
        self.data_layout.opacity = 1
        self.data_layout.disabled = False
        self.multi_face_images = []

if __name__ == "__main__":
    FaceRecKivyApp().run()
    