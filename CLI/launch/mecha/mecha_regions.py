# Valid GCP regions for c3-standard-176 Cloud Build worker pools
# 🤖 FULL MECHA FLEET: 18 regions globally!
# (Middle East regions excluded by user preference)

# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 MECHA C3 MANIFEST - SOURCE OF TRUTH FOR CLOUD BUILD REGIONS
# ═══════════════════════════════════════════════════════════════════════════════
# This is the canonical definition for MECHA C3 Cloud Build regions.
# Other modules should import from here, NOT from gcp-manifest.json
# gcp-manifest.json just points here as the source of truth.

MECHA_C3_MANIFEST = {
    "system": "mecha",
    "description": "MECHA Battle System - C3 Cloud Build worker pool regions",
    "registry_file": "CLI/launch/mecha/data/mecha_hangar.json",
    "machine_type_default": "c3-standard-176",
    "total_regions": 18,
    "regions": [
        # US Regions (8)
        {"code": "us-central1", "location": "Iowa, USA", "continent": "north_america", "latency": "low"},
        {"code": "us-east1", "location": "South Carolina, USA", "continent": "north_america", "latency": "low"},
        {"code": "us-east4", "location": "Northern Virginia, USA", "continent": "north_america", "latency": "low"},
        {"code": "us-east5", "location": "Columbus, Ohio, USA", "continent": "north_america", "latency": "low"},
        {"code": "us-west1", "location": "Oregon, USA", "continent": "north_america", "latency": "low"},
        {"code": "us-west2", "location": "Los Angeles, USA", "continent": "north_america", "latency": "low"},
        {"code": "us-west3", "location": "Salt Lake City, USA", "continent": "north_america", "latency": "low"},
        {"code": "us-west4", "location": "Las Vegas, USA", "continent": "north_america", "latency": "low"},
        # North America (1)
        {"code": "northamerica-northeast1", "location": "Montreal, Canada", "continent": "north_america", "latency": "low"},
        # Europe (5)
        {"code": "europe-west1", "location": "Belgium", "continent": "europe", "latency": "medium"},
        {"code": "europe-west2", "location": "London, UK", "continent": "europe", "latency": "medium"},
        {"code": "europe-west3", "location": "Frankfurt, Germany", "continent": "europe", "latency": "medium"},
        {"code": "europe-west4", "location": "Netherlands", "continent": "europe", "latency": "medium"},
        {"code": "europe-west9", "location": "Paris, France", "continent": "europe", "latency": "medium"},
        # Asia (2)
        {"code": "asia-northeast1", "location": "Tokyo, Japan", "continent": "asia", "latency": "high"},
        {"code": "asia-southeast1", "location": "Singapore", "continent": "asia", "latency": "high"},
        # Australia (1)
        {"code": "australia-southeast1", "location": "Sydney, Australia", "continent": "australia", "latency": "high"},
        # South America (1)
        {"code": "southamerica-east1", "location": "São Paulo, Brazil", "continent": "south_america", "latency": "medium-high"},
    ],
}

# Runtime lookup dict (derived from manifest - kept for backward compatibility)
C3_REGIONS = {
    # US Regions (7)
    "us-central1": {
        "location": "Iowa, USA",
        "zones": ["a", "b", "c", "f"],
        "latency_to_us": "low",
    },
    "us-east1": {
        "location": "South Carolina, USA",
        "zones": ["b", "c", "d"],
        "latency_to_us": "low",
    },
    "us-east4": {
        "location": "Northern Virginia, USA",
        "zones": ["a", "b", "c"],
        "latency_to_us": "low",
    },
    "us-east5": {
        "location": "Columbus, Ohio, USA",
        "zones": ["a", "b", "c"],
        "latency_to_us": "low",
    },
    "us-west1": {
        "location": "Oregon, USA",
        "zones": ["a", "b", "c"],
        "latency_to_us": "low",
    },
    "us-west2": {
        "location": "Los Angeles, USA",
        "zones": ["a", "b", "c"],
        "latency_to_us": "low",
    },
    "us-west3": {
        "location": "Salt Lake City, USA",
        "zones": ["a", "b", "c"],
        "latency_to_us": "low",
    },
    "us-west4": {
        "location": "Las Vegas, USA",
        "zones": ["a", "b", "c"],
        "latency_to_us": "low",
    },

    # North America (1)
    "northamerica-northeast1": {
        "location": "Montreal, Canada",
        "zones": ["a", "b", "c"],
        "latency_to_us": "low",
    },

    # Europe Regions (5)
    "europe-west1": {
        "location": "Belgium",
        "zones": ["b", "c", "d"],
        "latency_to_us": "medium",
    },
    "europe-west2": {
        "location": "London, UK",
        "zones": ["a", "b", "c"],
        "latency_to_us": "medium",
    },
    "europe-west3": {
        "location": "Frankfurt, Germany",
        "zones": ["a", "b", "c"],
        "latency_to_us": "medium",
    },
    "europe-west4": {
        "location": "Netherlands",
        "zones": ["a", "b", "c"],
        "latency_to_us": "medium",
    },
    "europe-west9": {
        "location": "Paris, France",
        "zones": ["a", "b", "c"],
        "latency_to_us": "medium",
    },

    # Asia Regions (2)
    "asia-northeast1": {
        "location": "Tokyo, Japan",
        "zones": ["a", "b", "c"],
        "latency_to_us": "high",
    },
    "asia-southeast1": {
        "location": "Singapore",
        "zones": ["a", "b", "c"],
        "latency_to_us": "high",
    },

    # Australia/Pacific (1)
    "australia-southeast1": {
        "location": "Sydney, Australia",
        "zones": ["a", "b", "c"],
        "latency_to_us": "high",
    },

    # South America (1)
    "southamerica-east1": {
        "location": "São Paulo, Brazil",
        "zones": ["a", "b", "c"],
        "latency_to_us": "medium-high",
    },
}

# Recommended region order (by latency to US)
US_PREFERRED_REGIONS = [
    "us-central1",               # Iowa (current)
    "us-east4",                  # Virginia
    "us-east1",                  # South Carolina
    "us-east5",                  # Ohio
    "us-west1",                  # Oregon
    "us-west2",                  # Los Angeles
    "us-west3",                  # Salt Lake City
    "us-west4",                  # Las Vegas
    "northamerica-northeast1",   # Montreal
]

EUROPE_PREFERRED_REGIONS = [
    "europe-west1",   # Belgium
    "europe-west2",   # London
    "europe-west3",   # Frankfurt
    "europe-west4",   # Netherlands
    "europe-west9",   # Paris
]

ASIA_PREFERRED_REGIONS = [
    "asia-northeast1",      # Tokyo
    "asia-southeast1",      # Singapore
    "australia-southeast1", # Sydney
]

SOUTH_AMERICA_REGIONS = [
    "southamerica-east1",   # São Paulo
]

ALL_REGIONS = list(C3_REGIONS.keys())

# Export simple list for MECHA system
ALL_MECHA_REGIONS = list(C3_REGIONS.keys())
