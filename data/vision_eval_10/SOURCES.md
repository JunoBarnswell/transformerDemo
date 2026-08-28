# vision_eval_10 数据来源与标注说明

用途：当前仓库 Detection Head 的 8/2 小样本开发训练与留出评测；不是生产数据集，也不代表 YOLOv8 对标结论。

来源仓库：https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection
上游数据说明：NEU Surface Defect Database；镜像 README 说明包含 6 类热轧钢表面缺陷、200x200 灰度图和 XML 框标注。
许可注意：镜像 README 未给出独立数据许可声明；本目录仅作本地研究/开发验证，未经数据权利人确认不得重新分发或用于商业用途。

类别映射：
- 0: crazing
- 1: inclusion
- 2: patches
- 3: pitted_surface
- 4: rolled-in_scale
- 5: scratches

拆分：8 张 train，2 张 test（inclusion_2、pitted_surface_2），测试样本类别在训练集中仍有同类图，但图片本身未参与训练。

每张图使用对应源 Pascal-VOC XML 的全部 bndbox，生成项目 manifest 和 annotated 预览；未使用 mask。
