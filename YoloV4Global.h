#ifndef YOLOV4_GLOBAL_H
#define YOLOV4_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(YOLOV4_LIBRARY)
#  define YOLOV4SHARED_EXPORT Q_DECL_EXPORT
#else
#  define YOLOV4SHARED_EXPORT Q_DECL_IMPORT
#endif

#endif // YOLOV3_GLOBAL_H
