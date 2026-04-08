#!/usr/bin/env python3
"""
Qwen3-VL API 调用脚本 - 使用 DashScope API 提取视觉元素

功能：分析图片并生成视觉描述（visual_description），包含风格、构图、画面元素等信息。
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dashscope import MultiModalConversation
    import dashscope
except ImportError:
    print("错误: 请先安装必要的库")
    print("运行: pip install dashscope")
    sys.exit(1)

from api_utils import setup_dashscope_url

# ============================================================================
# Prompt模板 - 引导Qwen3-VL提取详细的视觉信息
# ============================================================================

EXTRACTION_PROMPT = """Please analyze this image and provide a concise natural language description. Focus on the key visual elements:

**CRITICAL RULES - STRICTLY FOLLOW:**
- **ONLY describe what is VISIBLY present in the image** - Do NOT generate, imagine, or infer any content that is not directly visible
- **Do NOT add interpretations or assumptions** - Only describe what you can actually see
- **Use realistic, concrete, and descriptive language** - Avoid abstract or metaphorical descriptions
- **Be factual and objective** - Focus on observable visual facts, not subjective interpretations
- **Use English for all descriptions** - Write all descriptions in English, except when quoting or transcribing text content that appears in the image itself (e.g., OCR text in other languages should be preserved in its original language without translation, transliteration, or phonetic annotations like pinyin)

**1. VISUAL STYLE **
- Art style: Realistic, cartoon, anime, painting, photography, etc.
- Color palette: Dominant colors, color saturation, color temperature (warm/cool)
- Overall atmosphere: Mood, tone, emotional quality (only if clearly visible through visual cues)
- Image quality: Clarity, texture, level of detail, resolution appearance

**2. CAMERA/CINEMATOGRAPHY INFORMATION **
- Shot size: Extreme close-up, close-up, medium close-up, medium shot, medium long shot, long shot, extreme long shot, or wide shot
- Camera angle: Eye-level, high angle (looking down), low angle (looking up), bird's eye view, worm's eye view, Dutch angle, etc.
- Composition: Framing, rule of thirds, centered, etc.
- Depth of field: Shallow focus, deep focus, bokeh effects
- Camera position: Front view, side view, back view, etc.

**3. VISUAL ELEMENTS **
- Characters/People: Main subjects, their basic appearance and pose (only what is visible)
- Objects/Props: Key objects and their positions (only what is visible)
- Scene/Environment: Location type and basic setting (indoor/outdoor, based on visible elements)
- Lighting: Main lighting characteristics (as observable in the image)
- Spatial Relationships: Key relative positions between main elements (only for visible elements)

**4. TEXT/OCR RECOGNITION **
- Identify ALL readable text (subtitles, labels, signs, UI elements, watermarks, etc.)
- **CRITICAL: Do NOT translate, transliterate, or add phonetic annotations (e.g., pinyin) to any OCR text**
- **ONLY output the original text content as it appears in the image** - preserve the exact text without any modifications, translations, or annotations
- Note the position, font characteristics, and color of each text element
- Include both prominent and small/background text

Write a natural, coherent, and concise description following this order. Focus on the essential information for each category. Be brief and avoid overly detailed descriptions. **Remember: ONLY describe what is actually visible in the image - no speculation, no imagination, no abstract concepts that cannot be directly observed.**"""

# ============================================================================
# API 调用
# ============================================================================

def call_qwen_api(image_path: str, api_key: str, model: str = "qwen3-vl-plus", base_url: str = None) -> str:
    """
    调用 Qwen3-VL API 提取视觉元素
    
    Args:
        image_path: 图片路径
        api_key: API Key
        model: 模型名称
        base_url: API base URL（可选，默认使用 config 中 DASHSCOPE_BASE_URL）
        
    Returns:
        API 返回的文本内容
    """
    # 检查图片是否存在
    if not Path(image_path).exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    setup_dashscope_url(base_url)
    
    # 构建图片路径（使用 file:// 前缀）
    image_file_path = f"file://{image_path}"
    
    # 构建消息
    messages = [
        {
            'role': 'user',
            'content': [
                {'image': image_file_path},
                {'text': EXTRACTION_PROMPT}
            ]
        }
    ]
    
    # 调用 API
    print(f"📸 正在分析图片: {image_path}", flush=True)
    print("🤖 正在调用 Qwen3-VL API...", flush=True)
    sys.stdout.flush()
    
    try:
        response = MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages
        )
        
        # 检查响应状态
        if response.status_code != 200:
            raise Exception(f"API 调用失败: {response.message} (status_code: {response.status_code})")
        
        # 提取文本内容
        output_text = response.output.choices[0].message.content[0]["text"]
        print("✅ API 调用完成", flush=True)
        sys.stdout.flush()
        
        return output_text
        
    except Exception as e:
        print(f"❌ API 调用失败: {e}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        raise

# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL API 视觉元素提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 提取视觉元素并保存为JSON
  python qwen_vl_api.py --image first_frame.png --output elements.json
  
  # 提取并输出为文本格式
  python qwen_vl_api.py --image first_frame.png --output elements.txt --format text
  
  # 使用自定义模型和 API Key
  python qwen_vl_api.py --image first_frame.png --output elements.json --model "qwen3-vl-plus" --api-key "your-api-key"
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='输入图片路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径（.json 或 .txt）'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'text'],
        default=None,
        help='输出格式（不指定则根据 output 后缀推断）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型名称（默认: qwen3-vl-plus）'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='DashScope API Key（默认: 从环境变量 DASHSCOPE_API_KEY 读取）'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='API Base URL（默认: 使用 config 中 DASHSCOPE_BASE_URL）'
    )
    
    args = parser.parse_args()
    
    try:
        # 获取 API Key
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("❌ 错误: 未提供 API Key", file=sys.stderr)
            print("   请使用 --api-key 参数或设置环境变量 DASHSCOPE_API_KEY", file=sys.stderr)
            sys.exit(1)
        
        # 推断 format（根据 output 后缀）
        output_format = args.format
        if output_format is None:
            output_format = 'text' if str(args.output).endswith('.txt') else 'json'

        # 调用 API
        output_text = call_qwen_api(
            image_path=args.image,
            api_key=api_key,
            model=args.model or 'qwen3-vl-plus',
            base_url=args.base_url
        )
        
        result = {"visual_description": output_text.strip(), "image_path": args.image}
        
        # 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'json' or output_path.suffix == '.json':
            # JSON格式
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存到: {output_path} (JSON格式)", flush=True)
        else:
            # 文本格式
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['visual_description'])
            print(f"✅ 结果已保存到: {output_path} (文本格式)", flush=True)
        
        sys.stdout.flush()
        
        # 在控制台输出完整结果
        print("\n" + "=" * 80, flush=True)
        print("提取结果:", flush=True)
        print("=" * 80, flush=True)
        print(f"\n【视觉描述】", flush=True)
        print(result['visual_description'], flush=True)
        print("\n" + "=" * 80, flush=True)
        sys.stdout.flush()
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
