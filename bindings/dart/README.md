# Dart Bindings

这个目录包含了由 Flutter Rust Bridge 自动生成的 Dart 绑定代码。

## 文件说明

- `bridge_generated.dart`: 主要的绑定文件，包含所有 Rust API 的 Dart 接口
- `README.md`: 本文件

## 使用方法

1. 将这些文件复制到 Flutter 项目的 `lib/` 目录
2. 在 Dart 代码中导入：
   ```dart
   import 'bridge_generated.dart';
   ```

## 自动生成

这些文件会在构建时自动生成。如果修改了 `src/api.rs` 中的接口，请重新构建项目：

```bash
cd loci
cargo build
```

然后更新 Flutter 项目中的绑定文件。
