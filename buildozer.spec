[app]

# (str) عنوان التطبيق
title = نظام التعرف على الوجه

# (str) اسم الحزمة (package name)
package.name = facerecognition

# (str) نطاق الحزمة (domain)
package.domain = org.lazreg.face

# (str) دليل المصدر
source.dir = .

# (list) أنواع الملفات المضمنة
source.include_exts = py,png,jpg,kv,ttf,wav,mp3,gif

# (list) متطلبات التطبيق
requirements = python3,kivy,opencv,face_recognition,numpy,Pillow,arabic_reshaper,python-bidi

# (str) إصدار التطبيق
version = 1.0.0

# (str) وصف التطبيق
description = تطبيق التعرف على بصمة الوجه مع دعم اللغة العربية

# (list) أذونات Android
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,INTERNET

# (int) الحد الأدنى لإصدار Android
android.minapi = 21

# (int) الهدف من إصدار Android
android.api = 31

# (bool) تمكين التصحيح
android.debug = True

[buildozer]

# (int) مستوى السجل
log_level = 2

# (str) مسار ملفات البناء
bin_dir = ./bin