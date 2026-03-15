[app]

# (str) عنوان التطبيق
title = نظام التعرف على الوجه

# (str) اسم الحزمة
package.name = facerecognition

# (str) نطاق الحزمة
package.domain = org.lazreg.face

# (str) دليل المصدر
source.dir = .

# (list) أنواع الملفات المضمنة
source.include_exts = py,png,jpg,kv,ttf,wav,mp3,gif,txt

# (list) متطلبات التطبيق (مهم جداً)
requirements = python3,kivy==2.1.0,opencv-python-headless,face-recognition==1.3.0,numpy,Pillow,arabic_reshaper,python-bidi,requests

# (str) إصدار التطبيق
version = 1.0.0

# (str) وصف التطبيق
description = تطبيق التعرف على بصمة الوجه مع دعم اللغة العربية

# (str) الرخصة
license = MIT

# (list) أذونات Android (مهم جداً)
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,INTERNET,RECORD_AUDIO

# (int) الحد الأدنى لإصدار Android
android.minapi = 21

# (int) الهدف من إصدار Android
android.api = 31

# (bool) تمكين التصحيح
android.debug = True

# (bool) تمكين خدمات Google Play
android.gradle_dependencies = 'com.google.android.gms:play-services-vision:20.1.3'

# (str) لغة التطبيق الافتراضية
android.default_locale = ar

[buildozer]

# (int) مستوى السجل
log_level = 2

# (str) مسار ملفات البناء
bin_dir = ./bin