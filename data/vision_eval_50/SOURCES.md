# vision_eval_50 数据来源与标注说明

用途：六分类 Detection Head 的质量训练与独立留出评测。原始数据共 300 张，每类 50 张；每类 40 张 train、10 张 test。

来源仓库：https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection
来源树 SHA：351bee21b56bbf97a1e1d18262bceb202eb7ef41
上游数据：NEU Surface Defect Database；镜像 README 说明包含 6 类热轧钢表面缺陷、200x200 灰度图和 Pascal-VOC XML 框标注。
许可注意：镜像 README 未给出独立数据许可声明；本目录仅作本地研究/开发验证，未经数据权利人确认不得重新分发或用于商业用途。

类别映射：
- 0: crazing
- 1: inclusion
- 2: patches
- 3: pitted_surface
- 4: rolled-in_scale
- 5: scratches

标注规则：使用对应源 XML 的全部 bndbox，核验图像尺寸、类别和边界；生成的训练变体只允许对 train 图做几何/轻光照变换，并同步变换 bbox。
