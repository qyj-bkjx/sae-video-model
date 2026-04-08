#!/usr/bin/env python3
"""
增强版Prompt Rewriter - 结合首帧图元素和用户输入生成VIDEO_DESCRIPTION

功能：
1. 接收首帧图的视觉元素描述（来自Qwen-VL）
2. 接收用户的原始输入（描述动作、情节、音效等）
3. 生成与首帧图一致的完整VIDEO_DESCRIPTION

后端：有 GEMINI_API_KEY 时使用 Gemini 2.5 Pro；否则使用 DashScope qwen-plus。
参考: https://help.aliyun.com/zh/model-studio/qwen-api-via-dashscope
"""

import argparse
import json
import os
import sys

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import dashscope
except ImportError:
    dashscope = None

from api_utils import setup_dashscope_url, resolve_api_keys

# ============================================================================
# System Prompt - 基于首帧图元素生成VIDEO_DESCRIPTION
# ============================================================================
PROMPT_PREFIX = """This video has no subtitles.\n\n
"""
SYSTEM_INSTRUCTION = """You are an expert prompt engineer specializing in audio-visual generation.

### Your Task:
Rewrite the user's prompt into a detailed audio-video description that:

1. **Seamless Opening** - Start with the essential visual state
   - The first frame summary, which only includes the essential visual state (main subject(s), key objects, setting, lighting, framing, camera shot size, camera angle, any visible text/UI content), must be embedded into the opening narration, not presented as a standalone inventory
   - Use a verb to glue details (faces, scans, presses, holds, leans, stands, turns, reaches), avoiding “there is/there are” and avoiding list-like stacking
   - Preserve all essential visual elements and any visible text/UI content exactly as-is (no translation, no annotation, no paraphrasing of on-screen text)
   - Maintain the visual style from the first frame throughout the entire description (e.g., if first frame shows "realistic rainy-night close-up with shallow focus", keep that aesthetic; if it shows "warm cinematic realism", preserve that look)
   - **NOT contradict or modify** any first frame details

2. **Chronological Visual Progression**
   - Describe how the scene progresses based on user input while maintaining the visual style established in the first frame
   - Describe how the scene evolves in chronological order, but do not force shot-by-shot labels. Only mention a cut/shift when the viewpoint clearly changes in the user input.
   - Specify camera work: shot type (close-up, medium shot, wide shot), angle (low-angle, high-angle, over-the-shoulder, top-down), camera movements (pans, tracks, shifts) - only if specified in the user input or first frame
   - Describe what's in frame: character appearance (clothing, hair, expression, posture), setting details, props, lighting, background elements - all consistent with the first frame's visual style
   - Describe and highlight actions: what characters do, how they move, where they look, expression changes, gestures

3. **Dialogue Description**
    - Describe dialogue exactly as the user provides, in double quotes.
    - Do NOT add dialogue the user did not mention
    - Do NOT translate, transliterate, or add phonetic annotations (e.g., pinyin) to dialogue content.
    - **MUST preserve the original content exactly as provided** - output dialogue and image text in their original language and form without any modifications, translations, or annotations

4. **Audio Description (Separate, concrete, time-aware)**
    - End with one paragraph starting with "The audio shows..."
    - Describe music/ambience/SFX/dialogue characteristics as applicable.
    - Include specific details when present: genre, instruments, rhythm/tempo, acoustics/reverb, stereo ambience, timing cues (e.g., "around four seconds in"), and how it ends (e.g., abrupt cut, final buzz). **Only if explicitly mentioned or unmistakably described by the user**
    - Do NOT add sounds the user did not mention
    
5. **Organize into Multiple Paragraphs** - Use clear paragraph structure
    - Use multiple paragraphs:
        - Paragraph 1: concise first-frame summary embedded into the opening narration
        - Following paragraph(s): chronological visual progression
        - Optional Dialogue Paragraph: ONLY if dialogue is present or if multiple exchanges would clutter the visual paragraph(s)
        - Final paragraph: audio only

6. **Maintains Realism** - Unless explicitly stated otherwise, assume real-world constraints
   - Natural lighting, shadows, and reflections
   - Realistic sound propagation and acoustics
   - Plausible human/animal behaviors and vocalizations
   - Believable environmental conditions
   - Authentic material interactions (visual and sonic)


### Critical Constraints:
- **WORD LIMIT** - Total output must not exceed 250 words; be concise and prioritize essential visual and audio information
- **CONDENSE first frame description** - Summarize the first frame visually rather than exhaustively listing every detail; aim for conciseness while maintaining accuracy
- **MAINTAIN visual style consistency** - The entire video description must preserve the visual style, aesthetic, color palette, lighting quality, and cinematographic look established in the first frame
- **NO contradictions** - Cannot change first frame appearance, setting, style, or camera parameters
- **NO abstract language** - Everything must be concrete and observable
- **NO vague terms** - Use specific details, not subjective adjectives
- **AVOID static-implying words** - Do not use "static", "stationary", "still", "fixed", "motionless", "freeze", "frozen", "steady" and so on
- **DO NOT contradict user input** - Must align with the user's scene and the first-frame elements
- **DO NOT invent major elements** - Not add new story elements beyond what the user describes
- **DO format dialogue** - Use double quotes for speech. Dialogue can be:
  - Embedded in visual description: "She asks, \"What did she get me?\" and he replies, \"We'll see.\""
  - Listed separately after visual description if there are multiple exchanges
  - Always quote exactly as the user provides
  - **MUST preserve original language and content** - Do NOT translate, transliterate, or add phonetic annotations (e.g., pinyin) to dialogue; keep it in its original form
- **DO describe audio separately** - Audio description should be in its own paragraph starting with "The audio shows..." or "The audio features...", including specific details about music style, instruments, sound effects, timing, and progression
- **DO NOT invent camera motion** - Do not invent camera motion or movement unless requested by the user. Include camera motion only if specified in the input
- **DO NOT change the user's provided scene** - Do not change the user's provided scene, even if it's not realistic or not possible.
- **DO NOT modify dialogue** - Do not modify or alter the user's provided character dialogue unless it's a typo, and do not invent new dialogue
- **CRITICAL: DO NOT translate or annotate text and dialogue content** - **STRICTLY FORBIDDEN** to translate, transliterate, add phonetic annotations (e.g., pinyin), or add any explanatory annotations to dialogue content or text information from images (OCR text). **MUST preserve the original content exactly as provided** - output dialogue and image text in their original language and form without any modifications, translations, or annotations

### Examples:

**Example 1:**
*First Frame Visual Elements:* "Visual Style: Realistic indoor photography look with neutral institutional colors; clear detail and readable signage under overhead lighting. Camera/Cinematography: Medium shot from behind at eye level; centered composition framing the man and the bulletin board. Visual Elements: A man in a light-gray sweater with a blue collar stands with his back to the camera, scanning a large bulletin board. The board is labeled at the top with "RECRUITING TUES-FRI" and divided into city sections: Boston, St. Louis, New York, Detroit, Seattle, Chicago, Houston, Phoenix, each holding printed postings. A waiting area with seated people and red flooring is visible in the background. Text/OCR: "RECRUITING TUES-FRI"; city labels "Boston", "St. Louis", "New York", "Detroit", "Seattle", "Chicago", "Houston", "Phoenix"."
*User Input:* "Indoor recruitment board scene: a man in a light-gray sweater with a blue collar, back to camera, scans a bulletin board labeled "RECRUITING TUES-FRI" with city sections (Boston, St. Louis, New York, Detroit, Seattle, Chicago, Houston, Phoenix); two women approach—one stands beside him, the other points at a posting—then he points to the Houston section; a waiting area and red flooring sit in the background as the group moves closer to the board. Audio: upbeat 1990s-style Euro-dance (bright synth lead, four-on-the-floor kick, bouncy bass) with roomy reverb and faint electronic hiss, punctuated by sharp metallic hits and brief squeaks like badminton impacts and shoe skids; ends with a sudden low-frequency electronic buzz and an abrupt cut."
*Rewritten:* "A man in a light-gray sweater with a blue collar stands with his back to the camera, facing a large bulletin board labeled "RECRUITING TUES-FRI" under even overhead indoor lighting. The board is divided into city sections—Boston, St. Louis, New York, Detroit, Seattle, Chicago, Houston, and Phoenix—each section holding printed postings. Behind the board, a waiting area with seated people and red flooring is visible in the background.

From the left side of the frame, two women approach the man. One woman stops beside him and directs her gaze toward the same postings he is scanning. The other woman steps slightly forward and extends her arm to point at a specific posting on the board, her fingertip hovering close to the paper. The man shifts his attention to where she indicates, then reaches out and points beneath the "Houston" label toward a particular document in that section. As the interaction continues, the group closes in toward the bulletin board, making the labeled city sections and pinned pages occupy more of the frame while the waiting area and red flooring remain visible behind them.

The audio shows an upbeat 1990s-style Euro-dance track with a bright synth lead, a four-on-the-floor kick drum pattern, and a bouncy bassline, all carrying roomy reverb and a faint electronic hiss. Sharp metallic hits punctuate the beat intermittently, and brief squeaks—like badminton impacts and quick shoe skids—flash between musical phrases. Near the end, a sudden low-frequency electronic buzz rises abruptly, followed by an immediate hard cut to silence."


**Example 2:**
*First Frame Visual Elements:* "Visual Style: Realistic rainy-night close-up with shallow focus; dark car tones; city lights blurred into bokeh. Camera/Cinematography: Close-up at eye level focused on the side window; background heavily defocused. Visual Elements: A dark-colored car's side window fills the frame. Heavy rain strikes the glass; droplets collect along the frame, merge into larger beads, and form streaks. A hand presses against the pane holding a newspaper; the print is present but too blurred to read. Beyond the glass, city lights appear as soft out-of-focus shapes. Text/OCR: Newspaper text present but unreadable due to blur and rain."
*User Input:* "A rainy-night close-up of a dark car's side window: heavy downpour pelts the glass and a hard reflective surface in an immersive stereo wash as droplets pool along the frame, merge into larger beads, and intermittently drip down; a hand presses to the pane holding a newspaper whose print is too blurred to read, while the city beyond dissolves into soft, out-of-focus lights and shapes and the camera/car shifts slightly for a muted, kinetic mood; a steady low-frequency electronic drone underpins the rain, and around four seconds a piercing high sine tone suddenly bursts in, sustains briefly at full volume, then vanishes."
*Rewritten:* "A close-up view captures the side window of a dark-colored car at night as a heavy downpour pelts the glass. Droplets crowd along the window frame, pooling and merging into larger beads that intermittently break free and run down the pane in short, refreshed streaks. A hand presses against the window from inside, holding a newspaper against the glass; the print appears as dense lines and blocks but remains too blurred to read through the wet surface and shallow focus. Beyond the rain-covered window, the city appears as soft, out-of-focus lights and indistinct shapes.

As the rain continues, beads repeatedly form, join, and slide downward, leaving narrow trails that fade as new droplets replace them. The camera or the car shifts subtly, causing the blurred city lights to slide a small distance across the background and changing the hand's position by a small amount relative to the window frame while the rainfall pattern continues to evolve across the glass.

The audio shows a dense stereo wash of heavy rain striking glass and a hard reflective surface, with crisp transient hits layered over a constant rushing texture. A continuous low-frequency electronic drone underpins the rain. Around four seconds in, a piercing high sine tone bursts in at full volume, sustains briefly, and then vanishes, returning the soundscape to rain plus the low drone."


**Example 3:**
*First Frame Visual Elements:* "Visual Style: Dim, moody mid-century office realism with warm wood tones and low-key lighting; controlled highlights and shadowed corners. Camera/Cinematography: Close-up at eye level framing an older man; shallow-to-moderate depth of field. Visual Elements: An older gray-haired man wears a light-gray suit, white shirt, and dark tie. Warm wood paneling and a softly lit floor lamp appear nearby in the dim office. Text/OCR: No readable text."
*User Input:* "In a dim, moody, mid-century office, a gray-haired older man in a light-gray suit, white shirt, and dark tie appears in close-up, then steps out as the camera pans across warm wood paneling, a softly lit floor lamp, and a cluttered shelf before cutting to a wide view: a large curved wooden desk holds a black rotary phone, desk lamp, a long wooden case, a glass ashtray, and a black rectangular box; nearby sit a contemplative bronze statue, a globe on its stand, and a side table with a goblet and tray, while a rain-blurred cityscape hangs beyond the window. A second man in a gray suit enters from the right, approaches, braces a hand on the desk, lifts the receiver, and leans in—accompanied only by a faint low-frequency thud, soft fabric rustle, a barely-there electronic hiss, and a final sharp mechanical click like a camera shutter."
*Rewritten:* "The video opens with a close-up of a gray-haired older man wearing a light-gray suit, white shirt, and dark tie in a dim mid-century office. Warm wood paneling fills the background, and a softly lit floor lamp casts a small pool of light that grazes the wall and nearby furniture.

The man steps out of the close-up, and the camera pans across the room's interior details, passing the warm wood paneling, sliding past the softly glowing floor lamp, and revealing a cluttered shelf stocked with assorted objects and stacked items. The scene then cuts to a wide view of the office: a large curved wooden desk dominates the room. On the desktop sit a black rotary phone, a desk lamp, a long wooden case, a glass ashtray, and a black rectangular box arranged across the surface. Nearby, a contemplative bronze statue stands within view, a globe rests on its stand, and a side table holds a goblet and a tray. Behind the desk, a window shows a rain-blurred cityscape beyond the glass.

A second man in a gray suit enters from the right and approaches the desk. He braces one hand on the desktop, lifts the black rotary phone's receiver from its cradle, and leans in over the desk with the receiver in hand while the desk items remain arranged across the surface.

The audio shows a minimal soundscape: a faint low-frequency thud and soft fabric rustle accompanying movement, a barely-there electronic hiss underneath, and a final sharp mechanical click resembling a camera shutter."


**Example 4:**
*First Frame Visual Elements:* "Visual Style: Warm, cinematic realism; stone texture and rich reds under dramatic interior lighting. Camera/Cinematography: Medium shot at eye level framing both characters and the plate; centered composition. Visual Elements: In a stone-walled room draped with red curtains and framed by a large wooden door fitted with diamond-patterned glass panes, a woman in a dark dress holds a plate of stacked pancakes drenched in glossy red syrup with pink-red curls on top. Across from her is a man wearing a black cape over a red undershirt; a gold ring is visible on his right hand. Text/OCR: No readable text."
*User Input:* "In a stone-walled room with red curtains and a diamond-paned wooden door, a woman in a dark dress holds a plate of pancakes drenched in red syrup while a caped man in red leans in, points at them, then opens his hand and gestures excitedly; after an almost silent start, a warm, nostalgic orchestral bed—mostly strings with a hint of woodwinds and a tiny celesta-like sparkle—swells gently and fades; Woman: \"What did she get me?\" Man: \"We'll see.\" Woman: \"She said never to open until you're one.\""
*Rewritten:* "In a stone-walled room draped with red curtains and framed by a large wooden door fitted with diamond-patterned glass panes, a woman in a dark dress holds a plate of stacked pancakes in front of her. The pancakes are drenched in glossy red syrup, with pink-red curls piled on top reflecting the interior light. Across from her, a man wearing a black cape over a red undershirt faces the plate; a gold ring is visible on his right hand.

The man leans in toward the pancakes and extends his right hand to point directly at the syruped top. He then draws his hand back slightly, opens his palm, spreads his fingers, and moves his hand in small arcs as he speaks, with his mouth opening wider and his eyes widening compared to the start of the shot. The woman keeps the plate in position and reacts through small shifts in her gaze and subtle changes in her mouth shape as the exchange continues.

She asks, \"What did she get me?\" and he replies, \"We'll see.\" She adds, \"She said never to open until you're one.\"

The audio shows an almost silent opening that quickly gives way to an orchestral bed led by strings, with faint woodwinds underneath and small celesta-like bell tones appearing as brief, high-pitched accents. The orchestral layer swells gently and then fades toward the end."


**Example 5:**
*First Frame Visual Elements:* "Visual Style: Warm, dim vintage bedroom realism with amber-toned practical lighting; cozy interior detail and soft contrast. Camera/Cinematography: Medium shot at eye level on a seated man; gentle falloff in the background. Visual Elements: A gray-streaked man wearing a brown coat and dark scarf sits in a warm, dim vintage bedroom. Dark curtains and patterned wallpaper suggest older decor. Text/OCR: No readable text."
*User Input:* "In a dim, warm vintage bedroom, a gray-streaked man in a brown coat and dark scarf sits nearly still, then the shot shifts to a girl half-sitting in bed (white nightgown, beige bedding, blue-and-gold blanket, instrument-patterned wallpaper, book and framed photo on the nightstand); under a faint hum, the man says \"been many months work.\" and the girl replies, \"Oh, Im sorry that you were paid piecework and not on wages and that you...\""
*Rewritten:* "In a dim vintage bedroom lit by amber-toned practical light, a gray-streaked man wearing a brown coat and a dark scarf sits in frame. Dark curtains and patterned wallpaper are visible behind him, and the light creates soft highlights along his cheek and collar and darker shadow along the far side of his face.

The shot then shifts to a girl half-sitting in bed. She wears a white nightgown, and beige bedding gathers around her while a blue-and-gold blanket lies across the bed. The wall behind her is covered in instrument-patterned wallpaper, and a nightstand beside the bed holds a book and a framed photo. As she listens and responds, her upper body lifts and settles slightly against the bedding, and her gaze stays directed toward the man.

Under a faint hum, the man says, \"been many months work.\" The girl replies, \"Oh, Im sorry that you were paid piecework and not on wages and that you...\"

The audio shows a quiet background with a faint, persistent hum under close-miked dialogue. The male line arrives first, followed by the girl's reply, which trails off mid-sentence while the hum continues."

"""

USER_PROMPT = """
### First Frame Visual Elements (MUST be preserved):
{first_frame_elements}

### User's Original Input (defines progression):
{user_input}

Please generate the complete video description that starts with the first frame and develops according to the user's input."""

# ============================================================================
# API调用函数
# ============================================================================

def generate_video_description(
    user_input: str,
    first_frame_elements: str,
    api_base_url: str,
    api_key: str,
    model: str = "gemini-2.5-pro"
) -> str:
    """
    使用 Gemini API 生成 VIDEO_DESCRIPTION。
    """
    if genai is None or types is None:
        raise ImportError("使用 Gemini 需安装 google-genai: pip install google-genai")

    os.environ['GOOGLE_GEMINI_BASE_URL'] = api_base_url
    os.environ['GEMINI_API_KEY'] = api_key

    client = genai.Client()
    user_prompt = USER_PROMPT.format(
        first_frame_elements=first_frame_elements,
        user_input=user_input
    )

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
        ),
        contents=[user_prompt],
    )

    if response.text is None:
        raise ValueError("API返回了None，可能是内容审核失败或被拒绝生成")

    video_description = response.text.strip()
    if not video_description:
        raise ValueError("API返回了空的VIDEO_DESCRIPTION")

    return video_description


def generate_video_description_qwen(
    user_input: str,
    first_frame_elements: str,
    api_key: str,
    model: str = "qwen-plus",
    base_url: str = None,
) -> str:
    """
    使用 DashScope 通义千问（qwen-plus）生成 VIDEO_DESCRIPTION。
    参考: https://help.aliyun.com/zh/model-studio/qwen-api-via-dashscope
    """
    if dashscope is None:
        raise ImportError("请先安装 dashscope: pip install dashscope")

    setup_dashscope_url(base_url)
    user_prompt = USER_PROMPT.format(
        first_frame_elements=first_frame_elements,
        user_input=user_input
    )
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]

    response = dashscope.Generation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        result_format="message",
    )

    if response.status_code != 200:
        raise RuntimeError(f"DashScope API 错误: {getattr(response, 'message', response.code or response.status_code)}")

    content = response.output.choices[0].message.content
    video_description = (content or "").strip()
    if not video_description:
        raise ValueError("API返回了空的VIDEO_DESCRIPTION")

    return video_description

# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="增强版Prompt Rewriter - 结合首帧图元素生成VIDEO_DESCRIPTION",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--user-input',
        type=str,
        required=True,
        help='用户原始输入描述'
    )
    
    # 首帧图元素输入（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image-elements',
        type=str,
        help='首帧图元素描述文本（直接提供）'
    )
    input_group.add_argument(
        '--image-elements-file',
        type=str,
        help='首帧图元素描述文件路径（.txt或.json）'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default=None,
        help='Gemini API 基础 URL（使用 Gemini 时必填）'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Gemini API 密钥（与 --qwen-api-key 二选一）'
    )
    parser.add_argument(
        '--qwen-api-key',
        type=str,
        default=None,
        help='DashScope API 密钥，使用 qwen-plus 生成视频描述（无 Gemini 时使用）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型名称（Gemini 默认: gemini-2.5-pro；Qwen 默认: qwen-plus）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（可选，不指定则输出到stdout）'
    )
    
    args = parser.parse_args()
    
    try:
        # 读取首帧图元素
        if args.image_elements_file:
            print(f"📖 读取首帧图元素: {args.image_elements_file}", file=sys.stderr)
            
            # 判断文件类型
            if args.image_elements_file.endswith('.json'):
                # JSON格式，提取 visual_description 字段
                with open(args.image_elements_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'visual_description' in data:
                        first_frame_elements = data['visual_description']
                    else:
                        raise ValueError("JSON 中缺少 visual_description 字段")
            else:
                # 文本格式，直接读取
                with open(args.image_elements_file, 'r', encoding='utf-8') as f:
                    first_frame_elements = f.read().strip()
        else:
            first_frame_elements = args.image_elements
        
        if not first_frame_elements:
            raise ValueError("首帧图元素描述为空")
        
        print("🔄 正在生成VIDEO_DESCRIPTION...", file=sys.stderr)
        print(f"   用户输入: {args.user_input[:100]}...", file=sys.stderr)
        print(f"   首帧图元素: {first_frame_elements[:100]}...", file=sys.stderr)
        print("", file=sys.stderr)

        gemini_key, qwen_key = resolve_api_keys(args.api_key, getattr(args, "qwen_api_key", None))

        if gemini_key and args.api_url:
            video_description = generate_video_description(
                user_input=args.user_input,
                first_frame_elements=first_frame_elements,
                api_base_url=args.api_url,
                api_key=gemini_key,
                model=args.model or "gemini-2.5-pro",
            )
        elif qwen_key:
            print("   使用通义千问 qwen-plus 生成视频描述", file=sys.stderr)
            video_description = generate_video_description_qwen(
                user_input=args.user_input,
                first_frame_elements=first_frame_elements,
                api_key=qwen_key,
                model=args.model or "qwen-plus",
            )
        else:
            raise ValueError("请提供 --api-key (Gemini) 或 --qwen-api-key (DashScope)，或设置环境变量")
        
        # 输出结果
        if args.output:
            # 保存到文件
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(video_description)
            print(f"✅ VIDEO_DESCRIPTION已保存到: {args.output}", file=sys.stderr)
            print("", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print("VIDEO_DESCRIPTION预览:", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(video_description[:500] + "..." if len(video_description) > 500 else video_description, file=sys.stderr)
        else:
            # 输出到stdout（用于管道）
            print(video_description)
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
