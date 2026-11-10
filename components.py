"""Pure HTML rendering components for Corporate Strategy Consensus Engine.

All functions are pure: data in, sanitized HTML string out.
Strict compliance with html.escape() escaping and zero-emoji mandates.
"""

import html
from schemas import EvaluatedInitiative, StrategicHypothesis, VerdictEnum


def banner_html(buildings_b64: str | None) -> str:
    """Renders the top framed nocturnal banner at 320px with crisp pixel edges."""
    bg_style = (
        f"background-image: url('{buildings_b64}'); background-size: cover; "
        f"background-position: center bottom; background-repeat: no-repeat; "
        f"image-rendering: pixelated; image-rendering: crisp-edges;"
        if buildings_b64
        else "background-color: #050B16;"
    )

    return f"""
<div style="height: 320px; width: 100%; position: relative; overflow: hidden; border: 1px solid #1E4E68; margin-bottom: 24px; background-color: #050B16; {bg_style} box-sizing: border-box;">
    <div style="position: absolute; bottom: 20px; left: 20px; max-width: 520px; background: rgba(5, 9, 18, 0.94); border: 1px solid #E5A93C; padding: 14px 18px; box-shadow: 0 4px 16px rgba(0,0,0,0.9); z-index: 10;">
        <div style="font-family: 'Press Start 2P', monospace; font-size: 0.92rem; color: #F4A236; letter-spacing: 0.8px; line-height: 1.4;">Corporate Strategy Consensus Engine</div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #5CE1E6; letter-spacing: 0.8px; margin-top: 6px; line-height: 1.3;">Executive Decision Support & Multi-Agent Analysis</div>
    </div>
</div>
"""


def active_doc_card_html(active_name: str, active_hash: str, pointer_count: int) -> str:
    """Renders the Active Document status card in the Ingestion tab."""
    safe_name = html.escape(active_name or "Uploaded Document")
    safe_hash = html.escape(active_hash[:16])
    return f"""
<div style="background: rgba(6, 13, 24, 0.9); border: 1px solid #1E4E68; border-left: 3px solid #00F5D4; padding: 14px 18px; margin-bottom: 16px;">
    <div style="font-family: 'Press Start 2P', monospace; font-size: 0.72rem; color: #00F5D4; margin-bottom: 6px;">Active Document in Vault</div>
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #E0FBFC;"><b>File:</b> {safe_name} &nbsp;|&nbsp; <b>Hash:</b> <code>{safe_hash}...</code> &nbsp;|&nbsp; <b>Pointers:</b> {pointer_count}</div>
</div>
"""


def target_doc_badge_html(active_name: str, active_hash: str) -> str:
    """Renders the Target Document indicator in the Consensus tab."""
    safe_name = html.escape(active_name or "Corporate Strategic Plan")
    safe_hash = html.escape(active_hash[:12]) if active_hash else "None"
    return f"""
<div style="background: rgba(6, 13, 24, 0.7); border: 1px solid #1E4E68; padding: 10px 14px; margin-bottom: 14px;">
    <span style="font-family: 'Press Start 2P', monospace; font-size: 0.68rem; color: #5CE1E6;">Target Document:</span>
    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #E0FBFC; margin-left: 8px;">{safe_name} (<code>{safe_hash}...</code>)</span>
</div>
"""


def summary_html(executive_summary: str) -> str:
    """Renders the executive summary block."""
    safe_summary = html.escape(executive_summary)
    return f"""
<div style="background: rgba(6, 13, 24, 0.88); border: 1px solid #1B3B53; border-left: 3px solid #48CAE4; padding: 18px 22px; margin-bottom: 20px; box-sizing: border-box; width: 100%; overflow-wrap: break-word; word-break: break-word;">
    <div style="font-family: 'Press Start 2P', monospace; font-size: 0.78rem; color: #48CAE4; margin-bottom: 8px;">Executive Consensus Statement</div>
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; line-height: 1.6; color: #E0FBFC;">{safe_summary}</div>
</div>
"""


def section_tag_html(tag_text: str) -> str:
    """Renders a section header tag."""
    safe_text = html.escape(tag_text)
    return f"<div class='pixel-section-tag'>{safe_text}</div>"


def pixel_hr_html() -> str:
    """Renders the pixelated dashed separator line."""
    return "<hr class='pixel-hr'/>"


def initiative_card_html(
    init: EvaluatedInitiative,
    matched_hypothesis: StrategicHypothesis | None,
) -> str:
    """Renders an evaluated initiative card with full 4-quadrant layout."""
    title_text = (
        matched_hypothesis.title
        if matched_hypothesis and matched_hypothesis.title
        else "Strategic Initiative"
    )
    init_title = f"{init.initiative_id}: {title_text}"

    confidence = max(0.0, min(1.0, float(init.confidence_score)))
    pct = int(round(confidence * 100))
    filled_blocks = int(round(confidence * 10))
    empty_blocks = 10 - filled_blocks
    meter_ascii = f"[{'█' * filled_blocks}{'░' * empty_blocks}] {pct}%"

    if init.verdict == VerdictEnum.APPROVED:
        badge_bg = "#48CAE4"
        badge_text = "APPROVED"
    elif init.verdict == VerdictEnum.CONDITIONALLY_APPROVED:
        badge_bg = "#F4A236"
        badge_text = "CONDITIONALLY_APPROVED"
    else:
        badge_bg = "#FF4D6D"
        badge_text = "REJECTED"

    q1_text = (
        html.escape(matched_hypothesis.strategic_goal_alignment).replace("\n", "<br/>")
        if matched_hypothesis
        else "No goal alignment recorded in checkpoint."
    )
    q2_text = (
        html.escape(matched_hypothesis.execution_blueprint).replace("\n", "<br/>")
        if matched_hypothesis
        else "No execution blueprint recorded in checkpoint."
    )
    q3_text = (
        html.escape(matched_hypothesis.selection_rationale).replace("\n", "<br/>")
        if matched_hypothesis
        else "No selection rationale recorded in checkpoint."
    )

    pointer_html = ""
    if matched_hypothesis and matched_hypothesis.evidence_pointers:
        pills = "".join(
            f'<span class="pixel-ptr-pill">Pointer: {html.escape(pid)}</span>'
            for pid in matched_hypothesis.evidence_pointers
        )
        pointer_html = f"""
        <div style="margin-top: 10px; padding-top: 6px; border-top: 1px dashed #162E45;">
            <div style="font-family: 'VT323', monospace; font-size: 15px; color: #48CAE4; margin-bottom: 4px;">Evidence Citations:</div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px;">{pills}</div>
        </div>
        """

    q4_upside = (
        html.escape(matched_hypothesis.quantified_upside).replace("\n", "<br/>")
        if matched_hypothesis
        else "No projection metrics recorded."
    )
    q4_tradeoff = html.escape(init.trade_off_resolution)
    prereqs_list = "".join(
        f'<li style="margin-bottom: 3px;">{html.escape(p)}</li>'
        for p in init.mandatory_prerequisites
    )

    return f"""
<div style="background: rgba(6, 13, 24, 0.88); border: 1px solid #1B3B53; border-left: 3px solid #48CAE4; padding: 20px; margin-bottom: 16px; box-sizing: border-box; width: 100%; overflow-wrap: break-word; word-break: break-word; white-space: normal;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #162E45; padding-bottom: 10px; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
        <div style="font-family: 'Press Start 2P', monospace; font-size: 10px; color: #F4A236; letter-spacing: 0.5px;">{html.escape(init_title)}</div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 700; color: #5CE1E6;">{meter_ascii}</span>
            <span style="font-family: 'Press Start 2P', monospace; font-size: 9px; font-weight: 700; padding: 4px 8px; border: 1px solid #04070E; background: {badge_bg}; color: #04070E;">{badge_text}</span>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; box-sizing: border-box; width: 100%;">
        <div style="background: rgba(4, 8, 17, 0.7); border: 1px solid #162E45; padding: 12px; box-sizing: border-box; overflow-wrap: break-word; word-break: break-word;">
            <div style="font-family: 'VT323', monospace; font-size: 17px; color: #48CAE4; border-bottom: 1px dashed #162E45; padding-bottom: 4px; margin-bottom: 6px;">Strategic Goal Alignment</div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; line-height: 1.5; color: #E0FBFC;">{q1_text}</div>
        </div>
        <div style="background: rgba(4, 8, 17, 0.7); border: 1px solid #162E45; padding: 12px; box-sizing: border-box; overflow-wrap: break-word; word-break: break-word;">
            <div style="font-family: 'VT323', monospace; font-size: 17px; color: #48CAE4; border-bottom: 1px dashed #162E45; padding-bottom: 4px; margin-bottom: 6px;">Execution Blueprint</div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; line-height: 1.5; color: #E0FBFC;">{q2_text}</div>
        </div>
        <div style="background: rgba(4, 8, 17, 0.7); border: 1px solid #162E45; padding: 12px; box-sizing: border-box; overflow-wrap: break-word; word-break: break-word;">
            <div style="font-family: 'VT323', monospace; font-size: 17px; color: #48CAE4; border-bottom: 1px dashed #162E45; padding-bottom: 4px; margin-bottom: 6px;">Selection Rationale & Evidence</div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; line-height: 1.5; color: #E0FBFC;">
                {q3_text}
                {pointer_html}
            </div>
        </div>
        <div style="background: rgba(4, 8, 17, 0.7); border: 1px solid #162E45; padding: 12px; box-sizing: border-box; overflow-wrap: break-word; word-break: break-word;">
            <div style="font-family: 'VT323', monospace; font-size: 17px; color: #48CAE4; border-bottom: 1px dashed #162E45; padding-bottom: 4px; margin-bottom: 6px;">Quantified Upside & Arbiter Verdict</div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; line-height: 1.5; color: #E0FBFC;">
                <div style="margin-bottom: 8px;">
                    <strong style="color: #48CAE4; font-size: 10px; font-family: 'Press Start 2P', monospace; display: block; margin-bottom: 3px;">Projected Upside:</strong>
                    {q4_upside}
                </div>
                <div style="margin-top: 8px; padding: 6px 8px; background: #04070E; border-left: 3px solid #F4A236;">
                    <span style="font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 12px; color: #F4A236; display: block; margin-bottom: 2px;">Arbiter Trade-off Resolution:</span>
                    <span style="color: #F4A236;">{q4_tradeoff}</span>
                </div>
                <div style="margin-top: 8px; padding: 6px 8px; background: #04070E; border-left: 3px solid #FF4D6D;">
                    <span style="font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 12px; color: #FF4D6D; display: block; margin-bottom: 2px;">Mandatory Prerequisites:</span>
                    <ul style="margin: 0; padding-left: 16px; color: #F4A236;">{prereqs_list}</ul>
                </div>
            </div>
        </div>
    </div>
</div>
"""

