"""
Scenario Intermediate Representation (IR)

This module provides a structured IR for scenario descriptions that enables:
1. Deterministic geometric consistency validation
2. Clear feedback to the LLM about what was understood
3. Precise error messages with suggested fixes

The IR is extracted from free-form text using an LLM, then validated
using deterministic rules that the LLM cannot violate.
"""

import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# CARDINAL DIRECTIONS AND GEOMETRY
# =============================================================================

class Cardinal(Enum):
    """Cardinal directions."""
    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"
    UNKNOWN = "?"
    
    @classmethod
    def from_string(cls, s: str) -> "Cardinal":
        """Parse cardinal from string."""
        s = s.upper().strip()
        mapping = {
            "N": cls.NORTH, "NORTH": cls.NORTH,
            "S": cls.SOUTH, "SOUTH": cls.SOUTH,
            "E": cls.EAST, "EAST": cls.EAST,
            "W": cls.WEST, "WEST": cls.WEST,
        }
        return mapping.get(s, cls.UNKNOWN)
    
    def opposite(self) -> "Cardinal":
        """Get the opposite cardinal."""
        opposites = {
            Cardinal.NORTH: Cardinal.SOUTH,
            Cardinal.SOUTH: Cardinal.NORTH,
            Cardinal.EAST: Cardinal.WEST,
            Cardinal.WEST: Cardinal.EAST,
        }
        return opposites.get(self, Cardinal.UNKNOWN)
    
    def is_opposite(self, other: "Cardinal") -> bool:
        """Check if two cardinals are opposite."""
        return self.opposite() == other
    
    def is_perpendicular(self, other: "Cardinal") -> bool:
        """Check if two cardinals are perpendicular."""
        ns = {Cardinal.NORTH, Cardinal.SOUTH}
        ew = {Cardinal.EAST, Cardinal.WEST}
        if self in ns and other in ew:
            return True
        if self in ew and other in ns:
            return True
        return False
    
    def turn_result(self, maneuver: str) -> "Cardinal":
        """
        Given an approach cardinal and a maneuver, compute the exit cardinal.
        
        Convention: "approach" means where the vehicle is COMING FROM (heading).
        - Vehicle approaching from NORTH is heading SOUTH (going N→S)
        - If it turns LEFT, it exits heading EAST (going N→E, so exit is E)
        """
        if maneuver == "straight":
            return self.opposite()
        
        # Turn matrices based on where you're heading (opposite of approach)
        # If approaching from N (heading S), left turn → heading E
        left_turns = {
            Cardinal.NORTH: Cardinal.EAST,   # approaching from N, heading S, left → heading E
            Cardinal.SOUTH: Cardinal.WEST,   # approaching from S, heading N, left → heading W
            Cardinal.EAST: Cardinal.NORTH,   # approaching from E, heading W, left → heading N  
            Cardinal.WEST: Cardinal.SOUTH,   # approaching from W, heading E, left → heading S
        }
        right_turns = {
            Cardinal.NORTH: Cardinal.WEST,   # approaching from N, heading S, right → heading W
            Cardinal.SOUTH: Cardinal.EAST,   # approaching from S, heading N, right → heading E
            Cardinal.EAST: Cardinal.SOUTH,   # approaching from E, heading W, right → heading S
            Cardinal.WEST: Cardinal.NORTH,   # approaching from W, heading E, right → heading N
        }
        
        if maneuver == "left":
            return left_turns.get(self, Cardinal.UNKNOWN)
        elif maneuver == "right":
            return right_turns.get(self, Cardinal.UNKNOWN)
        
        return Cardinal.UNKNOWN


# =============================================================================
# IR DATACLASSES
# =============================================================================

@dataclass
class VehicleIR:
    """
    Intermediate representation of a vehicle's spatial configuration.
    
    IMPORTANT: We use the PIPELINE CONVENTION for approach_direction:
    - approach_direction = the direction the vehicle is HEADING/TRAVELING
    - NOT where the vehicle is coming from
    
    E.g., "approaches from the north heading south" → approach_direction = S (southbound)
    E.g., "traveling northward" → approach_direction = N (northbound)
    
    This matches the pipeline's constraint evaluation in csp.py.
    """
    vehicle_id: int
    approach_direction: Cardinal = Cardinal.UNKNOWN  # Direction of travel (heading)
    maneuver: str = "unknown"  # straight, left, right, lane_change
    exit_direction: Cardinal = Cardinal.UNKNOWN  # computed or explicit
    
    # Text spans that led to this extraction (for debugging)
    approach_text: str = ""
    maneuver_text: str = ""
    exit_text: str = ""
    
    def compute_exit(self) -> Cardinal:
        """
        Compute expected exit direction based on approach and maneuver.
        
        If traveling south (approach_direction=S) and turning left → exit east
        If traveling south (approach_direction=S) and turning right → exit west
        If traveling south (approach_direction=S) and straight → continue south
        """
        if self.approach_direction == Cardinal.UNKNOWN:
            return Cardinal.UNKNOWN
        if self.maneuver in ("unknown", "lane_change"):
            return Cardinal.UNKNOWN
        
        if self.maneuver == "straight":
            return self.approach_direction  # Continue in same direction
        
        # Left/right turns based on heading direction
        left_turns = {
            Cardinal.NORTH: Cardinal.WEST,   # heading N, left → heading W
            Cardinal.SOUTH: Cardinal.EAST,   # heading S, left → heading E
            Cardinal.EAST: Cardinal.NORTH,   # heading E, left → heading N
            Cardinal.WEST: Cardinal.SOUTH,   # heading W, left → heading S
        }
        right_turns = {
            Cardinal.NORTH: Cardinal.EAST,   # heading N, right → heading E
            Cardinal.SOUTH: Cardinal.WEST,   # heading S, right → heading W
            Cardinal.EAST: Cardinal.SOUTH,   # heading E, right → heading S
            Cardinal.WEST: Cardinal.NORTH,   # heading W, right → heading N
        }
        
        if self.maneuver == "left":
            return left_turns.get(self.approach_direction, Cardinal.UNKNOWN)
        elif self.maneuver == "right":
            return right_turns.get(self.approach_direction, Cardinal.UNKNOWN)
        
        return Cardinal.UNKNOWN
    
    def origin_cardinal(self) -> Cardinal:
        """
        The direction the vehicle is COMING FROM (opposite of approach_direction).
        If heading S (approach_direction=S), origin is N.
        """
        return self.approach_direction.opposite()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "approach_direction": self.approach_direction.value,  # heading direction (pipeline convention)
            "origin": self.origin_cardinal().value,  # where coming from
            "maneuver": self.maneuver,
            "exit": self.exit_direction.value if self.exit_direction != Cardinal.UNKNOWN else self.compute_exit().value,
        }


@dataclass  
class ConstraintIR:
    """
    Intermediate representation of an inter-vehicle constraint.
    """
    constraint_type: str  # follows_route_of, opposite_approach_of, perpendicular_left_of, etc.
    vehicle1_id: int
    vehicle2_id: int
    text_span: str = ""  # The exact text that triggered this extraction
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.constraint_type,
            "v1": self.vehicle1_id,
            "v2": self.vehicle2_id,
            "text": self.text_span,
        }


@dataclass
class ActorIR:
    """
    Intermediate representation of a non-ego actor.
    """
    actor_type: str  # parked_vehicle, walker, cyclist, static_prop
    description: str = ""
    affects_vehicle: Optional[int] = None
    position_text: str = ""
    motion_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.actor_type,
            "description": self.description,
            "affects_vehicle": self.affects_vehicle,
        }


@dataclass
class ScenarioIR:
    """
    Complete intermediate representation of a scenario.
    """
    vehicles: List[VehicleIR] = field(default_factory=list)
    constraints: List[ConstraintIR] = field(default_factory=list)
    actors: List[ActorIR] = field(default_factory=list)
    
    # Extraction metadata
    extraction_warnings: List[str] = field(default_factory=list)
    extraction_confidence: float = 1.0
    raw_llm_response: str = ""
    
    def get_vehicle(self, vehicle_id: int) -> Optional[VehicleIR]:
        """Get vehicle by ID."""
        for v in self.vehicles:
            if v.vehicle_id == vehicle_id:
                return v
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicles": [v.to_dict() for v in self.vehicles],
            "constraints": [c.to_dict() for c in self.constraints],
            "actors": [a.to_dict() for a in self.actors],
            "warnings": self.extraction_warnings,
            "confidence": self.extraction_confidence,
        }
    
    def to_summary_string(self) -> str:
        """Generate a human-readable summary for the critic prompt."""
        lines = ["Extracted IR:"]
        
        lines.append("  Vehicles:")
        for v in self.vehicles:
            origin = v.origin_cardinal().value
            heading = v.approach_direction.value
            expected_exit = v.compute_exit().value
            actual_exit = v.exit_direction.value if v.exit_direction != Cardinal.UNKNOWN else expected_exit
            lines.append(
                f"    Vehicle {v.vehicle_id}: heading {heading} (from {origin}), "
                f"{v.maneuver} → exits heading {actual_exit}"
            )
        
        if self.constraints:
            lines.append("  Constraints:")
            for c in self.constraints:
                lines.append(f"    {c.constraint_type}(V{c.vehicle1_id}, V{c.vehicle2_id})")
                if c.text_span:
                    lines.append(f"      from: \"{c.text_span}\"")
        
        if self.actors:
            lines.append("  Actors:")
            for a in self.actors:
                lines.append(f"    {a.actor_type}: {a.description}")
        
        if self.extraction_warnings:
            lines.append("  Warnings:")
            for w in self.extraction_warnings:
                lines.append(f"    [WARN] {w}")
        
        return "\n".join(lines)


# =============================================================================
# LLM-BASED IR EXTRACTION
# =============================================================================

IR_EXTRACTION_PROMPT = '''Extract structured information from this driving scenario as JSON.

CRITICAL: Output ONLY a single valid JSON object. No markdown, no code fences, no extra text before or after.

JSON SCHEMA (STRICT - follow exactly):
{
  "vehicles": [
    {"id": 1, "approach": "N/S/E/W or null", "approach_text": "exact text from scenario", "maneuver": "straight/left/right/lane_change or null", "maneuver_text": "exact text from scenario", "exit": "N/S/E/W or null"}
  ],
  "constraints": [
    {"type": "constraint_name", "v1": vehicle_id, "v2": vehicle_id, "text": "exact text from scenario"}
  ],
  "actors": [
    {"type": "parked_vehicle/walker/cyclist/static_prop", "description": "brief description", "affects_vehicle": vehicle_id or null}
  ],
  "warnings": ["list of any ambiguities or uncertainties"]
}

CONSTRAINT TYPES (use exact names):
- opposite_approach_of: vehicles heading opposite (N/S or E/W)
- same_approach_as: vehicles heading same direction
- perpendicular_left_of/perpendicular_right_of: perpendicular approach
- follows_route_of: V1 follows V2 (same heading, V1 behind)
- left_lane_of/right_lane_of: lane positions
- same_road_as/same_exit_as: shared paths
- merges_into_lane_of: lane change into another's lane

APPROACH = HEADING DIRECTION (where vehicle is GOING, NOT where it came from):
Examples:
- "from the north heading south" -> approach = "S"
- "traveling northward" -> approach = "N"
- "approaching from the east" with no direction -> approach = null

RULES:
1. All string values MUST be wrapped in double quotes
2. No trailing commas in arrays or objects
3. Use null (not None, not empty string) for missing values
4. Vehicle IDs must be integers
5. Keep text values exact - copy directly from scenario
6. If uncertain about a value, use null and add to warnings

SCENARIO:
{scenario_text}

Response (JSON only, no other text):'''

IR_REPAIR_PROMPT = '''The JSON you produced has a parsing error and is invalid.

Error: {error}

Return ONLY a valid JSON object that matches this exact schema. No markdown, no explanation, no text other than the JSON.

SCHEMA:
{
  "vehicles": [
    {"id": 1, "approach": "N/S/E/W or null", "approach_text": "text", "maneuver": "straight/left/right/lane_change or null", "maneuver_text": "text", "exit": "N/S/E/W or null"}
  ],
  "constraints": [
    {"type": "constraint_name", "v1": vehicle_id, "v2": vehicle_id, "text": "text"}
  ],
  "actors": [
    {"type": "parked_vehicle/walker/cyclist/static_prop", "description": "text", "affects_vehicle": vehicle_id or null}
  ],
  "warnings": ["text"]
}

CRITICAL JSON RULES:
1. All strings in double quotes (not single quotes)
2. No trailing commas
3. Use null for missing values (not empty strings or None)
4. Every object field must have a value
5. Numbers not in quotes, strings in quotes

SCENARIO:
{scenario_text}

PREVIOUS INVALID OUTPUT (analyze for patterns):
{bad_json}

CORRECTED JSON (output only JSON, no other text):'''

MAX_IR_REPAIR_SNIPPET = 3000

_NULLISH_STRINGS = {"", "null", "none", "n/a", "na", "unknown", "?"}


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        cleaned = "\n".join(lines)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def _truncate_for_prompt(text: str, limit: int = MAX_IR_REPAIR_SNIPPET) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def _extract_first_json_block(text: str) -> Optional[str]:
    """
    Extract the first balanced JSON object/array from text.
    This avoids greedy regex that can grab extra trailing content.
    """
    start = None
    open_ch = None
    close_ch = None
    for idx, ch in enumerate(text):
        if ch == "{":
            start = idx
            open_ch, close_ch = "{", "}"
            break
        if ch == "[":
            start = idx
            open_ch, close_ch = "[", "]"
            break
    if start is None:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _fix_trailing_commas(text: str) -> str:
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    return text


def _fix_common_json_issues(text: str) -> str:
    """Fix common JSON issues from LLM output."""
    # Fix trailing commas
    text = _fix_trailing_commas(text)

    # Remove stray backslashes before quotes outside of strings (e.g., \"key\": null).
    # This keeps escaped quotes inside strings intact.
    def _unescape_quotes_outside_strings(raw: str) -> str:
        out = []
        in_string = False
        escape = False
        i = 0
        while i < len(raw):
            ch = raw[i]
            if escape:
                out.append(ch)
                escape = False
                i += 1
                continue
            if ch == "\\":
                if not in_string and i + 1 < len(raw) and raw[i + 1] == '"':
                    i += 1
                    continue
                escape = True
                out.append(ch)
                i += 1
                continue
            if ch == '"':
                in_string = not in_string
            out.append(ch)
            i += 1
        return "".join(out)

    text = _unescape_quotes_outside_strings(text)

    # Fix escaped quotes around object keys (e.g., \"affects_vehicle\": null)
    text = re.sub(r'(?m)(^|[\s,{])\\\"([A-Za-z_][\w-]*)\\\"\\s*:', r'\1"\2":', text)
    
    # Fix unquoted keys (sometimes LLM forgets quotes)
    # This is a simple heuristic - replace bareword: with "bareword":
    text = re.sub(r'(?<=[{,\s])(\w+)\s*:', r'"\1":', text)
    
    # Fix single quotes used instead of double quotes (outside strings)
    # This is tricky - only do if no double quotes in text
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    
    # Fix Python-style None/True/False that slipped in
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)  
    text = re.sub(r'\bFalse\b', 'false', text)
    
    # Fix common escape issues: backslash before forward slashes in strings
    # Replace \/ with / (both are valid but LLM sometimes over-escapes)
    text = re.sub(r'\\/', '/', text)
    
    # Fix common newlines in JSON strings - ensure they're properly escaped
    # If we see a literal newline inside what should be a string, escape it
    # This is complex, so we'll use a simple approach: replace bare newlines with \n
    lines = text.split('\n')
    # Try to detect if we're in a string that got broken across lines
    fixed_lines = []
    in_string = False
    for line in lines:
        # Count unescaped quotes to track if we're in a string
        quote_count = 0
        escape = False
        for char in line:
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                quote_count += 1
        # If odd number of quotes, toggle in_string state
        if quote_count % 2 == 1:
            in_string = not in_string
        fixed_lines.append(line)
    text = '\n'.join(fixed_lines)
    
    # Remove any trailing garbage after the final } or ]
    # Find the position of the last closing brace/bracket
    last_brace = max(text.rfind('}'), text.rfind(']'))
    if last_brace > 0:
        text = text[:last_brace + 1]
    
    return text
    # Remove any trailing garbage after the JSON object
    # Find the last } and truncate
    depth = 0
    last_close = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_close = i
                break
    
    if last_close > 0 and last_close < len(text) - 1:
        text = text[:last_close + 1]
    
    return text


def _lenient_json_loads(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Try increasingly lenient JSON parsing strategies."""
    fixed = _fix_common_json_issues(text.strip())
    try:
        return json.loads(fixed), None
    except json.JSONDecodeError as e1:
        try:
            return json.loads(fixed, strict=False), (
                "Non-strict JSON parse used (control characters in strings)"
            )
        except json.JSONDecodeError as e2:
            # Fallback to Python literal evaluation after JSON->Python conversion
            py_text = re.sub(r'\bnull\b', 'None', fixed, flags=re.IGNORECASE)
            py_text = re.sub(r'\btrue\b', 'True', py_text, flags=re.IGNORECASE)
            py_text = re.sub(r'\bfalse\b', 'False', py_text, flags=re.IGNORECASE)
            try:
                return ast.literal_eval(py_text), "Used ast.literal_eval fallback"
            except (ValueError, SyntaxError) as e3:
                # Last resort: try to parse just the first JSON-like structure
                # Return detailed error for debugging
                return None, f"JSON parse error: {e1} (strict), {e2} (lenient), {e3} (ast)"


def _normalize_nullish(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _NULLISH_STRINGS:
        return None
    return value


def _coerce_vehicle_id(value: Any) -> Optional[int]:
    """Convert vehicle ID to integer. Handles numbers and letters (A=101, B=102, etc.)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        # First try to find a number
        match = re.search(r'\d+', value)
        if match:
            return int(match.group())
        # Handle letter IDs (A, B, C, etc.) - map to 101, 102, 103...
        # This avoids conflict with Vehicle 1, 2, 3
        if len(value) == 1 and value.upper().isalpha():
            return 100 + ord(value.upper()) - ord('A') + 1  # A=101, B=102, etc.
        # Handle "Vehicle A", "Vehicle B" format
        letter_match = re.search(r'vehicle\s*([A-Za-z])(?:\s|$)', value, re.IGNORECASE)
        if letter_match:
            letter = letter_match.group(1).upper()
            return 100 + ord(letter) - ord('A') + 1
    return None


def _normalize_maneuver(value: Any) -> str:
    value = _normalize_nullish(value)
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    if text in {"left_turn", "turn_left"}:
        return "left"
    if text in {"right_turn", "turn_right"}:
        return "right"
    if text in {"go_straight", "straight_ahead"}:
        return "straight"
    if text in {"lane_change", "lanechange", "change_lane", "lane_change_left", "lane_change_right"}:
        return "lane_change"
    return text


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def extract_ir_with_llm(
    scenario_text: str,
    generate_fn,  # Function that takes a prompt and returns LLM response
    max_repair_attempts: int = 1,
    debug: bool = False,
) -> ScenarioIR:
    """
    Extract structured IR from scenario text using an LLM.
    
    Args:
        scenario_text: The scenario description text
        generate_fn: A function that takes a prompt string and returns the LLM response
        max_repair_attempts: Maximum number of repair attempts for malformed JSON
        debug: Print debug information during extraction
        
    Returns:
        ScenarioIR with extracted information
    """
    # Avoid str.format interpreting the JSON braces in the prompt template.
    prompt = IR_EXTRACTION_PROMPT.replace("{scenario_text}", scenario_text)
    
    # DEBUG: Print prompt and response (only if debug flag is set)
    if debug:
        print(f"\n[IR_DEBUG] Sending prompt ({len(prompt)} chars) to LLM for IR extraction")
    
    response = generate_fn(prompt)
    
    if debug:
        print(f"[IR_DEBUG] LLM returned response: type={type(response).__name__}, len={len(response) if response else 0}")
        if response:
            print(f"[IR_DEBUG] Response: {response}")
        else:
            print(f"[IR_DEBUG] Response is None or empty!")
    
    def _attempt_parse(
        response_text: str,
    ) -> Tuple[Optional[Any], Optional[str], Optional[str], Optional[str]]:
        cleaned = _strip_code_fences(response_text)

        # Try to extract the first balanced JSON block from the response
        json_str = _extract_first_json_block(cleaned)
        if not json_str:
            # Fall back to greedy regex as a last resort
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                json_str = json_match.group()

        if not json_str:
            return None, None, None, "Failed to find JSON in LLM response"

        data, warning = _lenient_json_loads(json_str)
        if data is None:
            return None, json_str, None, warning or "JSON parse error"

        return data, json_str, warning, None

    # Parse JSON from response
    ir = ScenarioIR(raw_llm_response=response)

    try:
        data, json_str, warning, error = _attempt_parse(response)
        if warning:
            ir.extraction_warnings.append(warning)
        if data is None and max_repair_attempts > 0:
            for attempt in range(max_repair_attempts):
                ir.extraction_warnings.append(
                    f"IR repair attempt {attempt + 1}: {error}"
                )
                bad_json = json_str or response
                bad_json = _truncate_for_prompt(bad_json)
                repair_prompt = (
                    IR_REPAIR_PROMPT.replace("{scenario_text}", scenario_text)
                    .replace("{bad_json}", bad_json)
                    .replace("{error}", error or "JSON parse error")
                )
                if debug:
                    print(
                        f"\n[IR_DEBUG] Repair attempt {attempt + 1} "
                        f"({len(repair_prompt)} chars)"
                    )
                response = generate_fn(repair_prompt)
                ir.raw_llm_response = response
                data, json_str, warning, error = _attempt_parse(response)
                if warning:
                    ir.extraction_warnings.append(warning)
                if data is not None:
                    break

        if data is None:
            preview = response[:100] if len(response) > 100 else response
            attempt_note = (
                "after repairs" if max_repair_attempts > 0 else "without repairs"
            )
            ir.extraction_warnings.append(
                f"Failed to parse JSON {attempt_note}: {preview!r}"
            )
            ir.extraction_confidence = 0.0
            return ir

        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            ir.extraction_warnings.append("Parsed JSON is not an object")
            ir.extraction_confidence = 0.0
            return ir

        # Allow common singular/plural key variants
        if "vehicles" not in data and "vehicle" in data:
            data["vehicles"] = data["vehicle"]
        if "constraints" not in data and "constraint" in data:
            data["constraints"] = data["constraint"]
        if "actors" not in data and "actor" in data:
            data["actors"] = data["actor"]
        if "warnings" not in data and "warning" in data:
            data["warnings"] = data["warning"]
        
        # Parse vehicles
        # NOTE: "approach" in JSON = heading direction (where vehicle is GOING)
        # This matches the pipeline's approach_direction convention
        vehicles_data = data.get("vehicles", [])
        if isinstance(vehicles_data, dict):
            expanded = []
            for key, value in vehicles_data.items():
                if isinstance(value, dict):
                    if "id" not in value:
                        value = dict(value)
                        value["id"] = key
                    expanded.append(value)
                else:
                    expanded.append({"id": key})
            vehicles_data = expanded

        for v_data in vehicles_data:
            if not isinstance(v_data, dict):
                ir.extraction_warnings.append("Skipped non-object vehicle entry")
                continue
            vehicle_id = _coerce_vehicle_id(v_data.get("id"))
            if vehicle_id is None:
                ir.extraction_warnings.append(f"Skipped vehicle with invalid id: {v_data.get('id')!r}")
                continue

            approach_raw = _normalize_nullish(v_data.get("approach"))
            approach_str = str(approach_raw) if approach_raw is not None else "?"
            exit_raw = _normalize_nullish(v_data.get("exit"))
            exit_str = str(exit_raw) if exit_raw is not None else "?"

            vehicle = VehicleIR(
                vehicle_id=vehicle_id,
                approach_direction=Cardinal.from_string(approach_str or "?"),
                maneuver=_normalize_maneuver(v_data.get("maneuver")),
                exit_direction=Cardinal.from_string(exit_str or "?"),
                approach_text=_safe_text(v_data.get("approach_text")),
                maneuver_text=_safe_text(v_data.get("maneuver_text")),
                exit_text=_safe_text(v_data.get("exit_text")),
            )
            ir.vehicles.append(vehicle)
        
        # Parse constraints
        for c_data in data.get("constraints", []):
            if not isinstance(c_data, dict):
                ir.extraction_warnings.append("Skipped non-object constraint entry")
                continue
            v1 = _coerce_vehicle_id(c_data.get("v1"))
            v2 = _coerce_vehicle_id(c_data.get("v2"))
            if v1 is None or v2 is None:
                ir.extraction_warnings.append(
                    f"Skipped constraint with invalid ids: v1={c_data.get('v1')!r}, v2={c_data.get('v2')!r}"
                )
                continue
            constraint = ConstraintIR(
                constraint_type=_safe_text(c_data.get("type", "unknown")),
                vehicle1_id=v1,
                vehicle2_id=v2,
                text_span=_safe_text(c_data.get("text")),
            )
            ir.constraints.append(constraint)
        
        # Parse actors
        for a_data in data.get("actors", []):
            if not isinstance(a_data, dict):
                ir.extraction_warnings.append("Skipped non-object actor entry")
                continue
            actor = ActorIR(
                actor_type=_safe_text(a_data.get("type", "unknown")),
                description=_safe_text(a_data.get("description")),
                affects_vehicle=_coerce_vehicle_id(a_data.get("affects_vehicle")),
            )
            ir.actors.append(actor)
        
        # Add warnings from LLM
        warnings = data.get("warnings", [])
        if isinstance(warnings, str):
            warnings = [warnings]
        if isinstance(warnings, list):
            ir.extraction_warnings.extend(str(w) for w in warnings)
        
        # Validate vehicle IDs are present
        vehicle_ids = {v.vehicle_id for v in ir.vehicles}
        for c in ir.constraints:
            if c.vehicle1_id not in vehicle_ids:
                ir.extraction_warnings.append(f"Constraint references unknown Vehicle {c.vehicle1_id}")
            if c.vehicle2_id not in vehicle_ids:
                ir.extraction_warnings.append(f"Constraint references unknown Vehicle {c.vehicle2_id}")
        
        # Compute confidence based on completeness
        total_vehicles = len(ir.vehicles)
        known_approaches = sum(1 for v in ir.vehicles if v.approach_direction != Cardinal.UNKNOWN)
        known_maneuvers = sum(1 for v in ir.vehicles if v.maneuver not in ("unknown", None))
        
        if total_vehicles > 0:
            ir.extraction_confidence = (known_approaches + known_maneuvers) / (2 * total_vehicles)
        else:
            ir.extraction_confidence = 0.0
            ir.extraction_warnings.append("No vehicles extracted")
        
    except json.JSONDecodeError as e:
        ir.extraction_warnings.append(f"JSON parse error: {e}")
        ir.extraction_confidence = 0.0
    except KeyError as e:
        ir.extraction_warnings.append(f"Missing key in extracted data: {e}")
        ir.extraction_confidence = 0.0
    except Exception as e:
        ir.extraction_warnings.append(f"Extraction error: {type(e).__name__}: {e}")
        ir.extraction_confidence = 0.0
    
    return ir


def extract_ir_with_regex(scenario_text: str) -> ScenarioIR:
    """
    Fast regex-based IR extraction for simple cases.
    Falls back to this when LLM extraction fails or for quick validation.
    
    NOTE: approach_direction = HEADING direction (where vehicle is GOING)
    "approaches from the north heading south" → approach_direction = S (southbound)
    """
    ir = ScenarioIR()
    text_lower = scenario_text.lower()
    
    # Split into sentences to avoid cross-sentence matching
    # Simple sentence boundary detection
    sentences = re.split(r'[.!?]+', text_lower)
    
    # Extract vehicles
    vehicle_pattern = r'vehicle\s*(\d+)'
    vehicle_ids = sorted(set(int(m) for m in re.findall(vehicle_pattern, text_lower)))
    
    for vid in vehicle_ids:
        vehicle = VehicleIR(vehicle_id=vid)
        
        # Find sentences mentioning this vehicle
        vehicle_sentences = [s for s in sentences if re.search(rf'\bvehicle\s*{vid}\b', s)]
        vehicle_text = ' '.join(vehicle_sentences)
        
        # FIRST priority: Look for explicit heading direction (this IS the approach_direction)
        heading_patterns = [
            # "heading south", "traveling north", "moving west" - within sentence
            (rf'vehicle\s*{vid}\s+[^.]*?heading\s+(?:to(?:wards?)?\s+(?:the\s+)?)?(north|south|east|west)', 1),
            (rf'vehicle\s*{vid}\s+[^.]*?traveling\s+(?:to(?:wards?)?\s+(?:the\s+)?)?(north|south|east|west)', 1),
            (rf'vehicle\s*{vid}\s+[^.]*?moving\s+(?:to(?:wards?)?\s+(?:the\s+)?)?(north|south|east|west)', 1),
            # "northbound", "southward"
            (rf'vehicle\s*{vid}\s+[^.]*?(north|south|east|west)(?:ward|bound)', 1),
            # "toward the west", "towards the north" - strict matching
            (rf'vehicle\s*{vid}\s+[^.]*?to(?:wards?)?\s+(?:the\s+)?(north|south|east|west)', 1),
        ]
        for pattern, group in heading_patterns:
            match = re.search(pattern, vehicle_text)
            if match:
                vehicle.approach_direction = Cardinal.from_string(match.group(group))
                vehicle.approach_text = match.group(0)
                break
        
        # SECOND priority: If heading not found, look for "from" + origin (infer heading as opposite)
        if vehicle.approach_direction is None:
            origin_patterns = [
                # "approaches from the north" → origin N → heading S
                (rf'vehicle\s*{vid}\s+(?:approaches|enters|comes|arrives)\s+from\s+(?:the\s+)?(north|south|east|west)', 1),
                (rf'vehicle\s*{vid}\s+[^.]*?from\s+(?:the\s+)?(north|south|east|west)', 1),
            ]
            for pattern, group in origin_patterns:
                match = re.search(pattern, vehicle_text)
                if match:
                    origin = Cardinal.from_string(match.group(group))
                    # Origin is opposite of heading, so heading = opposite of origin
                    vehicle.approach_direction = origin.opposite()
                    vehicle.approach_text = match.group(0)
                    break
        
        # Try to find maneuver
        maneuver_patterns = [
            (rf'vehicle\s*{vid}\s+[^.]*?(?:turn(?:s|ing)?|make(?:s|ing)?)\s+(?:a\s+)?(left|right)', "turn"),
            (rf'vehicle\s*{vid}\s+[^.]*?(?:intends?\s+to\s+)?turn\s+(left|right)', "turn"),
            (rf'vehicle\s*{vid}\s+[^.]*?(left|right)(?:-hand)?\s+turn', "turn"),
            (rf'vehicle\s*{vid}\s+[^.]*?(?:go(?:es|ing)?|continu(?:e|es|ing)?)\s+straight', "straight"),
            (rf'vehicle\s*{vid}\s+[^.]*?lane\s+chang', "lane_change"),
            (rf'vehicle\s*{vid}\s+[^.]*?chang(?:e|es|ing)\s+lanes?', "lane_change"),
        ]
        for pattern, maneuver_type in maneuver_patterns:
            match = re.search(pattern, vehicle_text)
            if match:
                if maneuver_type == "turn":
                    vehicle.maneuver = match.group(1)  # left or right
                else:
                    vehicle.maneuver = maneuver_type
                vehicle.maneuver_text = match.group(0)
                break
        
        ir.vehicles.append(vehicle)
    
    # Extract constraints
    # Note: Use more precise patterns that don't span multiple sentences
    constraint_patterns = [
        # follows_route_of patterns
        (r'(vehicle\s*\d+)\s+(?:is\s+)?(?:follows?|following)\s+(?:the\s+)?(?:route\s+(?:of\s+)?|behind\s+)(vehicle\s*\d+)', "follows_route_of"),
        (r'(vehicle\s*\d+)\s+follows_route_of\s+(vehicle\s*\d+)', "follows_route_of"),
        
        # opposite_approach_of patterns
        (r'(vehicle\s*\d+)\s+(?:is\s+)?opposite_approach_of\s+(vehicle\s*\d+)', "opposite_approach_of"),
        (r'(vehicle\s*\d+)\s+(?:approaches?\s+from\s+(?:the\s+)?)?opposite\s+(?:approach|direction)\s+(?:of\s+|as\s+|to\s+)(vehicle\s*\d+)', "opposite_approach_of"),
        
        # perpendicular patterns - more precise to avoid over-matching
        (r'(vehicle\s*\d+)\s+(?:is\s+)?perpendicular_right_of\s+(vehicle\s*\d+)', "perpendicular_right_of"),
        (r'(vehicle\s*\d+)\s+(?:is\s+)?perpendicular_left_of\s+(vehicle\s*\d+)', "perpendicular_left_of"),
        
        # same_approach_as patterns  
        (r'(vehicle\s*\d+)\s+(?:is\s+)?same_approach_as\s+(vehicle\s*\d+)', "same_approach_as"),
        (r'(vehicle\s*\d+)\s+(?:approaches?\s+from\s+(?:the\s+)?)?same\s+(?:approach|direction)\s+(?:as\s+)(vehicle\s*\d+)', "same_approach_as"),
        
        # lane patterns
        (r'(vehicle\s*\d+)\s+(?:is\s+)?(?:in\s+)?(?:the\s+)?left_lane_of\s+(vehicle\s*\d+)', "left_lane_of"),
        (r'(vehicle\s*\d+)\s+(?:is\s+)?(?:in\s+)?(?:the\s+)?left\s+lane\s+(?:of\s+)(vehicle\s*\d+)', "left_lane_of"),
        (r'(vehicle\s*\d+)\s+(?:is\s+)?(?:in\s+)?(?:the\s+)?right_lane_of\s+(vehicle\s*\d+)', "right_lane_of"),
        (r'(vehicle\s*\d+)\s+(?:is\s+)?(?:in\s+)?(?:the\s+)?right\s+lane\s+(?:of\s+)(vehicle\s*\d+)', "right_lane_of"),
        
        # merge patterns
        (r'(vehicle\s*\d+)\s+(?:is\s+)?merges_into_lane_of\s+(vehicle\s*\d+)', "merges_into_lane_of"),
        (r'(vehicle\s*\d+)\s+merges?\s+into\s+(?:the\s+)?lane\s+(?:of\s+)(vehicle\s*\d+)', "merges_into_lane_of"),
        
        # same_road_as patterns
        (r'(vehicle\s*\d+)\s+(?:is\s+)?same_road_as\s+(vehicle\s*\d+)', "same_road_as"),
        (r'(vehicle\s*\d+)\s+(?:is\s+)?(?:on\s+(?:the\s+)?)?same\s+road\s+(?:as\s+)(vehicle\s*\d+)', "same_road_as"),
    ]
    
    for pattern, constraint_type in constraint_patterns:
        for match in re.finditer(pattern, text_lower):
            v1_match = re.search(r'\d+', match.group(1))
            v2_match = re.search(r'\d+', match.group(2))
            if v1_match and v2_match:
                constraint = ConstraintIR(
                    constraint_type=constraint_type,
                    vehicle1_id=int(v1_match.group()),
                    vehicle2_id=int(v2_match.group()),
                    text_span=match.group(0),
                )
                # Avoid duplicates
                if not any(c.constraint_type == constraint.constraint_type and 
                          c.vehicle1_id == constraint.vehicle1_id and 
                          c.vehicle2_id == constraint.vehicle2_id for c in ir.constraints):
                    ir.constraints.append(constraint)
    
    # Extract actors
    actor_patterns = [
        (r'parked\s*(?:_|\s)*vehicle', "parked_vehicle"),
        (r'pedestrian|walker', "walker"),
        (r'cyclist|bicycle|bike', "cyclist"),
        (r'traffic\s+cone|barrier|debris|obstacle', "static_prop"),
    ]
    
    for pattern, actor_type in actor_patterns:
        if re.search(pattern, text_lower):
            ir.actors.append(ActorIR(actor_type=actor_type, description=pattern))
    
    # Compute confidence
    total_vehicles = len(ir.vehicles)
    if total_vehicles > 0:
        known_approaches = sum(1 for v in ir.vehicles if v.approach_direction != Cardinal.UNKNOWN)
        known_maneuvers = sum(1 for v in ir.vehicles if v.maneuver not in ("unknown", None))
        ir.extraction_confidence = (known_approaches + known_maneuvers) / (2 * total_vehicles)
    
    return ir


def merge_ir_extractions(llm_ir: ScenarioIR, regex_ir: ScenarioIR) -> ScenarioIR:
    """
    Merge LLM and regex extractions, preferring LLM but filling gaps with regex.
    """
    merged = ScenarioIR()
    merged.extraction_confidence = max(llm_ir.extraction_confidence, regex_ir.extraction_confidence)
    
    # Use LLM vehicles as base, fill gaps from regex
    llm_vehicle_ids = {v.vehicle_id for v in llm_ir.vehicles}
    regex_vehicle_ids = {v.vehicle_id for v in regex_ir.vehicles}
    
    for v in llm_ir.vehicles:
        merged.vehicles.append(v)
    
    # Add regex vehicles that LLM missed
    for v in regex_ir.vehicles:
        if v.vehicle_id not in llm_vehicle_ids:
            merged.vehicles.append(v)
            merged.extraction_warnings.append(f"Vehicle {v.vehicle_id} found by regex but not LLM")
    
    # Merge constraints (prefer LLM)
    constraint_keys = set()
    for c in llm_ir.constraints:
        key = (c.constraint_type, c.vehicle1_id, c.vehicle2_id)
        constraint_keys.add(key)
        merged.constraints.append(c)
    
    for c in regex_ir.constraints:
        key = (c.constraint_type, c.vehicle1_id, c.vehicle2_id)
        if key not in constraint_keys:
            merged.constraints.append(c)
            merged.extraction_warnings.append(f"Constraint {c.constraint_type} found by regex but not LLM")
    
    # Merge actors (prefer LLM)
    merged.actors = llm_ir.actors if llm_ir.actors else regex_ir.actors
    
    # Combine warnings
    merged.extraction_warnings.extend(llm_ir.extraction_warnings)
    
    return merged
