# IC-Light

IC-Light 是一个控制图像照明的项目。

“IC-Light”这个名称代表“Impose Constant Light”（我们将在本页末尾简要描述这一点）。

目前，我们发布了两种类型的模型：文本条件重新光照模型和背景条件模型。两种类型都将前景图像作为输入。

# 开始使用

请按下面命令依次运行

    git clone https://github.com/lllyasviel/IC-Light.git
    cd IC-Light
    conda create -n iclight python=3.10
    conda activate iclight
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python gradio_demo.py

使用背景运行

    python gradio_demo_bg.py

模行会自动下载



# 实列


---

**Prompt: beautiful woman, detailed face, warm atmosphere, at home, bedroom**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/87265483-aa26-4d2e-897d-b58892f5fdd7)

---

**Prompt: beautiful woman, detailed face, sunshine from window**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/148c4a6d-82e7-4e3a-bf44-5c9a24538afc)

---

**beautiful woman, detailed face, neon, Wong Kar-wai, warm**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/f53c9de2-534a-42f4-8272-6d16a021fc01)

---

**Prompt: beautiful woman, detailed face, sunshine, outdoor, warm atmosphere**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/25d6ea24-a736-4a0b-b42d-700fe8b2101e)

---

**Prompt: beautiful woman, detailed face, sunshine, outdoor, warm atmosphere**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/dd30387b-0490-46ee-b688-2191fb752e68)

---

**Prompt: beautiful woman, detailed face, sunshine from window**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/6c9511ca-f97f-401a-85f3-92b4442000e3)

---

**Prompt: beautiful woman, detailed face, shadow from window**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/e73701d5-890e-4b15-91ee-97f16ea3c450)

---

**Prompt: beautiful woman, detailed face, sunset over sea**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/ff26ac3d-1b12-4447-b51f-73f7a5122a05)

---

**Prompt: handsome boy, detailed face, neon light, city**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/d7795e02-46f7-444f-93e7-4d6460840437)

---

**Prompt: beautiful woman, detailed face, light and shadow**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/706f70a8-d1a0-4e0b-b3ac-804e8e231c0f)

(beautiful woman, detailed face, soft studio lighting)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fe0a72df-69d4-4e11-b661-fb8b84d0274d)

---

**Prompt: Buddha, detailed face, sci-fi RGB glowing, cyberpunk**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/68d60c68-ce23-4902-939e-11629ccaf39a)

---

**Prompt: Buddha, detailed face, natural lighting**

Lighting Preference: Left

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/1841d23d-0a0d-420b-a5ab-302da9c47c17)

---

**Prompt: toy, detailed face, shadow from window**

Lighting Preference: Bottom

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/dcb97439-ea6b-483e-8e68-cf5d320368c7)

---

**Prompt: toy, detailed face, sunset over sea**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/4f78b897-621d-4527-afa7-78d62c576100)

---

**Prompt: dog, magic lit, sci-fi RGB glowing, studio lighting**

Lighting Preference: Bottom

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/1db9cac9-8d3f-4f40-82e2-e3b0cafd8613)

---

**Prompt: mysteriou human, warm atmosphere, warm atmosphere, at home, bedroom**

Lighting Preference: Right

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/5d5aa7e5-8cbd-4e1f-9f27-2ecc3c30563a)

---

### 自定背景条件模型

背景条件模型不需要很明确的提示。只需使用“帅哥、电影灯光”等简单提示即可。

---

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/0b2a889f-682b-4393-b1ec-2cabaa182010)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/477ca348-bd47-46ff-81e6-0ffc3d05feb2)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/5bc9d8d9-02cd-442e-a75c-193f115f2ad8)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/a35e4c57-e199-40e2-893b-cb1c549612a9)

---

# 说明

在 HDR 空间中，照明具有所有光传输都是独立的属性。

因此，不同光源的外观混合相当于混合光源的外观：

![cons](https://github.com/lllyasviel/IC-Light/assets/19834515/27c67787-998e-469f-862f-047344e100cd)

以上面的灯光阶段为例，来自“外观混合”和“光源混合”的两个图像是一致的（理想情况下，在 HDR 空间中数学上等效）。
![r2](https://github.com/lllyasviel/IC-Light/assets/19834515/25068f6a-f945-4929-a3d6-e8a152472223)

从左到右依次是输入、模型输出、重新照明、分割的阴影图像和合并的法线贴图。请注意，该模型未使用任何法线贴图数据进行训练。这个正常的估计来自于重新点亮的一致性。

您可以使用此按钮重现此实验（速度慢 4 倍，因为它重新点亮图像 4 次）

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/d9c37bf7-2136-446c-a9a5-5a341e4906de)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fcf5dd55-0309-4e8e-9721-d55931ea77f0)

以下是更大的图像（请随意尝试以获得更多结果！）

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/12335218-186b-4c61-b43a-79aea9df8b21)

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/2daab276-fdfa-4b0c-abcb-e591f575598a)

For reference, [geowizard](https://fuxiao0719.github.io/projects/geowizard/) (geowizard is a really great work!):

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/4ba1a96d-e218-42ab-83ae-a7918d56ee5f)

And, [switchlight](https://arxiv.org/pdf/2402.18848) (switchlight is another great work!):

![image](https://github.com/lllyasviel/IC-Light/assets/19834515/fbdd961f-0b26-45d2-802e-ffd734affab8)

# 模型说明

* **iclight_sd15_fc.safetensors** -  默认的重新照明模型，以文本和前景为条件。您可以使用初始潜伏来影响重新照明。

* **iclight_sd15_fcon.safetensors** - 与“iclight_sd15_fc.safetensors”相同，但使用偏移噪声进行训练。请注意，在用户研究中，默认的“iclight_sd15_fc.safetensors”稍微优于此模型。这就是为什么默认模型是没有偏移噪声的模型的原因。

* **iclight_sd15_fbc.safetensors** -  以文本、前景和背景为条件的重新照明模型。

# 引用

    @Misc{iclight,
      author = {Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
      title  = {IC-Light GitHub Page},
      year   = {2024},
    }

# 相关工作

Also read ...

[Total Relighting: Learning to Relight Portraits for Background Replacement](https://augmentedperception.github.io/total_relighting/)

[Relightful Harmonization: Lighting-aware Portrait Background Replacement](https://arxiv.org/abs/2312.06886)

[SwitchLight: Co-design of Physics-driven Architecture and Pre-training Framework for Human Portrait Relighting](https://arxiv.org/pdf/2402.18848)
