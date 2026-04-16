#!/usr/bin/env python3
"""
Revised Comprehensive Semantic Shift Words Database
This list is curated to include words that are more likely to appear in 
historical texts (pre-1950) while also having documented semantic shifts.
"""

# This is a simplified, direct dictionary for our pipeline.
# The complex class structure has been removed for clarity and directness.

COMPREHENSIVE_SEMANTIC_SHIFTS = {
    # === WORDS WITH HIGH HISTORICAL PREVALENCE ===

    "broadcast": {
        "senses": ["to_scatter_seed", "telecommunications"],
        "transition": (1920, 1940),
                "markers": {
            "to_scatter_seed": ["seed", "sow", "scatter", "field", "farm", "agriculture"],
            "telecommunications": ["radio", "television", "program", "signal", "station", "news"]
        }
    },
    "cell": {
        "senses": ["monastic_room", "biological_unit", "mobile_phone"],
        "transition": (1860, 1990), # Double transition
                "markers": {
            "monastic_room": ["monk", "monastery", "prison", "room", "small"],
            "biological_unit": ["biology", "organism", "membrane", "nucleus", "dna"],
            "mobile_phone": ["phone", "call", "mobile", "device", "tower", "text"]
        }
    },
    "engine": {
        "senses": ["mechanical_device", "software_component"],
                "transition": (1990, 2000),
                "markers": {
            "mechanical_device": ["steam", "machine", "piston", "motor", "vehicle", "locomotive"],
            "software_component": ["game", "graphics", "software", "search", "recommendation", "code"]
        }
    },
            "gay": {
                "senses": ["happy_cheerful", "homosexual_identity"],
        "transition": (1950, 1970),
                "markers": {
            "happy_cheerful": ["happy", "cheerful", "joyful", "merry", "bright", "lighthearted"],
            "homosexual_identity": ["homosexual", "community", "pride", "lgbtq", "orientation", "identity"]
        }
    },
    "nice": {
        "senses": ["foolish_silly", "pleasant_agreeable"],
        "transition": (1700, 1800), # This shift is early, but the word is very common
                "markers": {
            "foolish_silly": ["silly", "foolish", "stupid", "ignorant", "trivial"],
            "pleasant_agreeable": ["pleasant", "agreeable", "kind", "good", "lovely", "charming"]
        }

            },
            "awful": {
                "senses": ["inspiring_awe", "terrible_bad"],
                "transition": (1800, 1850),
                "markers": {
            "inspiring_awe": ["awe", "reverence", "majestic", "divine", "solemn", "impressive"],
            "terrible_bad": ["terrible", "bad", "horrible", "unpleasant", "dreadful", "nasty"]
        }
    },
    "call": {
        "senses": ["shout_summon", "telephone_communication"],
        "transition": (1880, 1920),
        "markers": {
            "shout_summon": ["shout", "cry", "name", "summon", "visit", "herald"],
            "telephone_communication": ["phone", "telephone", "ring", "number", "line", "call"]
        }
    },
     "line": {
        "senses": ["rope_or_thread", "ancestry_or_lineage", "telephone_connection"],
                "transition": (1880, 1920),
                "markers": {
            "rope_or_thread": ["rope", "thread", "string", "draw", "mark"],
            "ancestry_or_lineage": ["family", "ancestry", "descendants", "bloodline", "heritage"],
            "telephone_connection": ["phone", "telephone", "call", "busy", "connection"]
        }
    },
    "press": {
        "senses": ["to_push_firmly", "journalism_or_media"],
        "transition": (1800, 1850),
        "markers": {
            "to_push_firmly": ["push", "squeeze", "weight", "force", "compress"],
            "journalism_or_media": ["news", "journalism", "media", "newspaper", "publish", "freedom"]
        }
    },
    "port": {
        "senses": ["a_harbor_for_ships", "a_computer_interface"],
        "transition": (1980, 1990),
        "markers": {
            "a_harbor_for_ships": ["ship", "harbor", "boat", "dock", "sea", "trade"],
            "a_computer_interface": ["computer", "usb", "serial", "parallel", "connect", "device"]
        }
    },
    "power": {
        "senses": ["authority_or_influence", "electrical_energy"],
        "transition": (1880, 1910),
        "markers": {
            "authority_or_influence": ["king", "queen", "government", "control", "strength", "authority"],
            "electrical_energy": ["electricity", "energy", "grid", "station", "supply", "electric"]
        }
    },

    # --- Less likely but still possible candidates ---
    "cool": {
        "senses": ["temperature", "fashionable"],
        "transition": (1940, 1960),
                "markers": {
            "temperature": ["cold", "chilly", "weather", "calm", "refreshing"],
            "fashionable": ["awesome", "great", "hip", "trendy", "neat"]
        }
            },
            "pilot": {
                "senses": ["ship_guide", "aircraft_operator"],
                "transition": (1900, 1920),
                "markers": {
            "ship_guide": ["ship", "harbor", "boat", "navigate", "sea"],
            "aircraft_operator": ["plane", "fly", "aircraft", "flight", "aviation"]
        }
    }
}

if __name__ == "__main__":
    # Demo the comprehensive semantic shifts database
    shifts = COMPREHENSIVE_SEMANTIC_SHIFTS
    
    print("=== COMPREHENSIVE SEMANTIC SHIFTS DATABASE ===")
    for word, data in shifts.items():
        print(f"{word}: {data['senses']} ({data['transition']})") 