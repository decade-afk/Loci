/**
 * Loci Flutter FFI 绑定
 * 提供 Dart 侧的 Loci 引擎接口
 */

import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

/// 错误码常量
class LociErrorCode {
  static const int ok = 0;
  static const int invalidHandle = -1;
  static const int nullPointer = -2;
  static const int initFailed = -3;
  static const int generationFailed = -4;
}

/// 流式回调类型
typedef LociStreamCallbackNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Void> userData,
  ffi.Pointer<ffi.Char> token,
  ffi.Int32 tokenLen,
);
typedef LociStreamCallbackDart = int Function(
  ffi.Pointer<ffi.Void> userData,
  ffi.Pointer<ffi.Char> token,
  int tokenLen,
);

/// C FFI 函数签名定义
typedef LociInitNative = ffi.Pointer<ffi.Void> Function(
  ffi.Pointer<ffi.Char> modelPath,
  ffi.Int32 nThreads,
  ffi.Int32 nGpuLayers,
);
typedef LociInitDart = ffi.Pointer<ffi.Void> Function(
  ffi.Pointer<ffi.Char> modelPath,
  int nThreads,
  int nGpuLayers,
);

typedef LociGenerateNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Void> engine,
  ffi.Pointer<ffi.Char> prompt,
  ffi.Int32 maxTokens,
  ffi.Pointer<ffi.Char> outText,
  ffi.Int32 outLen,
);
typedef LociGenerateDart = int Function(
  ffi.Pointer<ffi.Void> engine,
  ffi.Pointer<ffi.Char> prompt,
  int maxTokens,
  ffi.Pointer<ffi.Char> outText,
  int outLen,
);

typedef LociDestroyNative = ffi.Int32 Function(ffi.Pointer<ffi.Void> engine);
typedef LociDestroyDart = int Function(ffi.Pointer<ffi.Void> engine);

/// Loci 引擎 Flutter 封装
class LociEngine {
  late ffi.DynamicLibrary _dylib;
  late LociInitDart _init;
  late LociGenerateDart _generate;
  late LociDestroyDart _destroy;

  ffi.Pointer<ffi.Void>? _handle;

  LociEngine() {
    // 根据平台加载动态库
    if (Platform.isAndroid) {
      _dylib = ffi.DynamicLibrary.open('libloci.so');
    } else if (Platform.isIOS) {
      _dylib = ffi.DynamicLibrary.process(); // iOS 静态链接
    } else if (Platform.isMacOS) {
      _dylib = ffi.DynamicLibrary.open('libloci.dylib');
    } else if (Platform.isLinux) {
      _dylib = ffi.DynamicLibrary.open('libloci.so');
    } else if (Platform.isWindows) {
      _dylib = ffi.DynamicLibrary.open('loci.dll');
    } else {
      throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
    }

    // 绑定 C 函数
    _init = _dylib.lookupFunction<LociInitNative, LociInitDart>('loci_init');
    _generate = _dylib.lookupFunction<LociGenerateNative, LociGenerateDart>('loci_generate');
    _destroy = _dylib.lookupFunction<LociDestroyNative, LociDestroyDart>('loci_destroy');
  }

  /// 初始化引擎
  ///
  /// [modelPath]: GGUF 模型文件路径
  /// [nThreads]: CPU 线程数（-1 表示自动检测）
  /// [nGpuLayers]: GPU 层数（-1 表示全部，0 表示纯 CPU）
  ///
  /// 返回 true 表示成功，false 表示失败
  bool init(String modelPath, {int nThreads = -1, int nGpuLayers = -1}) {
    final pathPtr = modelPath.toNativeUtf8();
    try {
      _handle = _init(pathPtr.cast(), nThreads, nGpuLayers);
      return _handle != ffi.nullptr;
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// 生成文本（同步阻塞）
  ///
  /// [prompt]: 输入提示词
  /// [maxTokens]: 最大生成 token 数
  ///
  /// 返回生成的文本，失败时返回 null
  String? generate(String prompt, {int maxTokens = 100}) {
    if (_handle == null || _handle == ffi.nullptr) {
      throw StateError('Engine not initialized. Call init() first.');
    }

    final promptPtr = prompt.toNativeUtf8();
    final outputPtr = malloc.allocate<ffi.Char>(8192);

    try {
      final result = _generate(
        _handle!,
        promptPtr.cast(),
        maxTokens,
        outputPtr,
        8192,
      );

      if (result == LociErrorCode.ok) {
        return outputPtr.cast<Utf8>().toDartString();
      } else {
        print('Loci generation failed with error code: $result');
        return null;
      }
    } finally {
      malloc.free(promptPtr);
      malloc.free(outputPtr);
    }
  }

  /// 销毁引擎，释放资源
  void dispose() {
    if (_handle != null && _handle != ffi.nullptr) {
      _destroy(_handle!);
      _handle = null;
    }
  }
}

/// 扩展方法：将 Dart String 转换为 C char*
extension StringToNativeUtf8 on String {
  ffi.Pointer<Utf8> toNativeUtf8() {
    return toNativeUtf8Pointer(this);
  }
}

ffi.Pointer<Utf8> toNativeUtf8Pointer(String str) {
  final units = str.codeUnits;
  final result = malloc.allocate<ffi.Uint8>(units.length + 1);
  final nativeString = result.asTypedList(units.length + 1);
  nativeString.setAll(0, units);
  nativeString[units.length] = 0; // null terminator
  return result.cast();
}
