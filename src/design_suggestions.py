"""
Saree Design Suggestion Engine.

Extracts dominant colours from a captured textile frame and returns
curated design suggestions (pattern, motif, colour combinations)
tailored to the detected palette.
"""

import cv2
import numpy as np
from collections import Counter


# ── Colour-to-name mapping (basic HSV-based classifier) ─────────────

def _hsv_to_colour_name(h, s, v):
    """Map an HSV triplet to a human-readable colour name."""
    if v < 40:
        return "Black"
    if s < 30 and v > 200:
        return "White"
    if s < 40:
        return "Grey"

    # Hue ranges (OpenCV uses 0-179)
    if h < 10 or h >= 170:
        return "Red"
    if 10 <= h < 25:
        return "Orange"
    if 25 <= h < 35:
        return "Yellow"
    if 35 <= h < 80:
        return "Green"
    if 80 <= h < 100:
        return "Teal"
    if 100 <= h < 130:
        return "Blue"
    if 130 <= h < 150:
        return "Purple"
    if 150 <= h < 170:
        return "Pink"
    return "Unknown"


# ── Design database keyed by primary colour ──────────────────────────

DESIGN_DATABASE = {
    "Red": {
        "palettes": [
            {"name": "Royal Crimson", "colors": ["#DC143C", "#FFD700", "#8B0000"], "description": "Deep crimson base with gold zari accents — a classic Banarasi look."},
            {"name": "Bridal Maroon", "colors": ["#800020", "#FFE4C4", "#C0392B"], "description": "Rich maroon paired with bisque borders, ideal for wedding ceremonies."},
            {"name": "Vermilion Sunset", "colors": ["#E34234", "#FF6347", "#FFA07A"], "description": "Warm vermilion gradients with light salmon highlights."},
        ],
        "patterns": ["Paisley Jaal", "Temple Border", "Floral Butis", "Zari Chevron"],
        "motifs": ["Lotus", "Mango (Ambi)", "Peacock", "Kalka"],
    },
    "Orange": {
        "palettes": [
            {"name": "Saffron Silk", "colors": ["#FF9933", "#FFFFF0", "#DAA520"], "description": "Vibrant saffron with ivory and goldenrod, echoing traditional Kanjivaram."},
            {"name": "Sunset Spice", "colors": ["#FF7F50", "#FFA500", "#8B4513"], "description": "Coral-orange blended with warm saddle-brown borders."},
        ],
        "patterns": ["Checks (Kattam)", "Coin Dots (Pottu)", "Diagonal Stripes"],
        "motifs": ["Sun Disc", "Rudraksha Chain", "Mango Cluster"],
    },
    "Yellow": {
        "palettes": [
            {"name": "Turmeric Gold", "colors": ["#FFD700", "#FFFACD", "#B8860B"], "description": "Classic turmeric gold with lemon chiffon and dark goldenrod border."},
            {"name": "Lemon Zest", "colors": ["#FFF44F", "#FAFAD2", "#DAA520"], "description": "Bright lemon with subtle golden zari weave."},
        ],
        "patterns": ["Running Vine Border", "Small Butis Grid", "Geometric Diamond"],
        "motifs": ["Jasmine Chain", "Sunflower", "Parrot Pair"],
    },
    "Green": {
        "palettes": [
            {"name": "Emerald Pattu", "colors": ["#50C878", "#006400", "#FFD700"], "description": "Lush emerald with dark green pallu and gold zari."},
            {"name": "Forest Silk", "colors": ["#228B22", "#2E8B57", "#FFDAB9"], "description": "Deep forest green with sea-green shading and peach-puff contrast border."},
            {"name": "Parrot Green", "colors": ["#6BBE45", "#32CD32", "#FFD700"], "description": "Bright parrot-green body with lime accents and gold motifs."},
        ],
        "patterns": ["Leaf Creeper", "Banana Leaf Border", "Diamond Jaal"],
        "motifs": ["Parrot", "Lotus Bud", "Mango (Ambi)", "Elephant Procession"],
    },
    "Blue": {
        "palettes": [
            {"name": "Indigo Dream", "colors": ["#4B0082", "#191970", "#C0C0C0"], "description": "Deep indigo with midnight blue and silver threadwork."},
            {"name": "Royal Navy", "colors": ["#000080", "#4169E1", "#FFD700"], "description": "Navy base with royal-blue gradient and gold zari pallu."},
            {"name": "Sky Pashmina", "colors": ["#87CEEB", "#B0E0E6", "#FFFFFF"], "description": "Soft sky-blue with powder blue accents, light and elegant."},
        ],
        "patterns": ["Wave Stripe", "Peacock Eye Jaal", "Cloud Motif Border"],
        "motifs": ["Peacock", "Fish (Matsya)", "Cloud Scroll"],
    },
    "Purple": {
        "palettes": [
            {"name": "Amethyst Elegance", "colors": ["#9B59B6", "#8E44AD", "#F1C40F"], "description": "Regal amethyst with contrasting golden border."},
            {"name": "Lavender Grace", "colors": ["#E6E6FA", "#9370DB", "#DDA0DD"], "description": "Soft lavender with medium-purple accents and plum borders."},
        ],
        "patterns": ["Diagonal Stripes", "Floral Mesh", "Star Jaal"],
        "motifs": ["Lotus", "Grapevine", "Crescent Moon"],
    },
    "Pink": {
        "palettes": [
            {"name": "Rose Petal", "colors": ["#FF69B4", "#FFB6C1", "#C71585"], "description": "Hot pink with light-pink shading and deep medium-violet-red border."},
            {"name": "Magenta Royal", "colors": ["#FF00FF", "#DA70D6", "#FFD700"], "description": "Bold magenta with orchid tones and golden zari."},
        ],
        "patterns": ["Floral Butis", "Heart Motif Border", "Scalloped Edge"],
        "motifs": ["Rose", "Butterfly", "Parrot Pair"],
    },
    "White": {
        "palettes": [
            {"name": "Ivory Kasavu", "colors": ["#FFFFF0", "#FFD700", "#DAA520"], "description": "Classic Kerala-style ivory with golden kasavu border."},
            {"name": "Pearl Tissue", "colors": ["#FAFAFA", "#C0C0C0", "#B0C4DE"], "description": "Off-white tissue with silver and steel-blue accents."},
        ],
        "patterns": ["Simple Border", "Thin Zari Stripe", "Minimalist Butis"],
        "motifs": ["Jasmine", "Lotus", "Temple Arch"],
    },
    "Black": {
        "palettes": [
            {"name": "Midnight Zari", "colors": ["#000000", "#FFD700", "#8B0000"], "description": "Deep black with luxurious gold zari and dark-red accents."},
            {"name": "Noir Elegance", "colors": ["#1C1C1C", "#C0C0C0", "#708090"], "description": "Jet black with silver and slate-grey threadwork."},
        ],
        "patterns": ["Geometric Blocks", "Abstract Mesh", "Bold Stripe"],
        "motifs": ["Peacock Feather", "Star", "Abstract Eye"],
    },
    "Grey": {
        "palettes": [
            {"name": "Silver Mist", "colors": ["#C0C0C0", "#808080", "#FFD700"], "description": "Elegant silver-grey with gold highlights."},
            {"name": "Ash Silk", "colors": ["#A9A9A9", "#696969", "#FF6347"], "description": "Ash grey with tomato-red contrast border."},
        ],
        "patterns": ["Herringbone Weave", "Micro Checks", "Pin Stripe"],
        "motifs": ["Geometric Star", "Abstract Vine", "Coin Dot"],
    },
    "Teal": {
        "palettes": [
            {"name": "Teal Tradition", "colors": ["#008080", "#20B2AA", "#FFD700"], "description": "Rich teal with light sea-green and gold zari border."},
            {"name": "Ocean Silk", "colors": ["#2F4F4F", "#5F9EA0", "#FFDEAD"], "description": "Dark slate teal with cadet-blue shading and navajo-white border."},
        ],
        "patterns": ["Wave Stripe", "Diamond Jaal", "Leaf Vine"],
        "motifs": ["Fish (Matsya)", "Conch Shell", "Lotus"],
    },
}


def extract_dominant_colours(frame, k=5):
    """
    Extract the top-k dominant colours from a BGR frame.
    Returns a list of dicts: [{name, hex, percentage}, …]
    """
    # Resize for speed
    small = cv2.resize(frame, (100, 100))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Flatten pixels
    pixels_hsv = hsv.reshape(-1, 3)

    # Map every pixel to a colour name
    names = [_hsv_to_colour_name(int(p[0]), int(p[1]), int(p[2])) for p in pixels_hsv]

    # Count and rank
    counter = Counter(names)
    total = len(names)
    results = []
    for colour_name, count in counter.most_common(k):
        pct = round((count / total) * 100, 1)
        results.append({
            "name": colour_name,
            "percentage": pct,
        })
    return results


def get_design_suggestions(frame):
    """
    Main entry: given a BGR frame, return dominant colours + design suggestions.
    """
    colours = extract_dominant_colours(frame, k=5)
    primary = colours[0]["name"] if colours else "White"

    suggestions = DESIGN_DATABASE.get(primary, DESIGN_DATABASE["White"])

    return {
        "dominant_colours": colours,
        "primary_colour": primary,
        "suggested_palettes": suggestions["palettes"],
        "suggested_patterns": suggestions["patterns"],
        "suggested_motifs": suggestions["motifs"],
    }
