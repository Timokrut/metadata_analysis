CAMERA_TAGS = {
    "Make",
    "Model",
    "LensModel",
    "LensInfo",
    "ISO",
    "ExposureTime",
    "FNumber",
    "FocalLength",
    "Flash",
    "WhiteBalance",
    "MeteringMode",
    "ExposureMode"
}

CAMERA_KEY_MARKERS = {
    "make", "model", "lensmodel", "lensinfo",
    "iso", "exposuretime", "fnumber", "focallength",
    "flash", "whitebalance", "meteringmode", "exposuremode"
}

CAMERA_TAG_PATTERNS = [
    r'\bmake\b', r'\bmodel\b', r'\blensmodel\b', r'\blensinfo\b',
    r'\biso\b', r'\bexposuretime\b', r'\bfnumber\b', r'\baperture\b',
    r'\bfocallength\b', r'\bflash\b', r'\bwhitebalance\b',
    r'\bmeteringmode\b', r'\bexposuremode\b'
]

GPS_TAGS = {
    "GPSLatitude",
    "GPSLongitude",
    "GPSAltitude",
    "GPSDateStamp"
}

SCREENSHOT_TAGS = {
    "BitDepth",
    "ColorType",
    "Compression",
    "Filter",
    "Interlace",
    "PixelsPerUnitX",
    "PixelsPerUnitY"
}

AI_TAGS = {
    "Prompt",
    "NegativePrompt",
    "Sampler",
    "Steps",
    "Seed",
    "CFGScale",
    "Workflow",
    "ModelHash"
}

AI_SOFTWARE = {
    "stable diffusion",
    "automatic1111",
    "comfyui",
    "fooocus",
    "invokeai",
    "midjourney",
    "runway",
    "flux",
    "dall-e",
    "firefly"
}

AI_VIDEO_SOFTWARE = {
    "runway", "gen-2", "gen-3", "pika", "pika labs",
    "sora", "openai", "stable video diffusion", "svd",
    "modelscope", "zeroscope", "vidgen", "moonvalley",
    "kling", "hailuo", "luma"
}

RICHNESS_TAGS = {
    "Make",
    "Model",
    "LensModel",
    "ISO",
    "ExposureTime",
    "FNumber",
    "DateTimeOriginal",
    "CreateDate"
}

VIDEO_RICHNESS_TAGS = {
    "duration", "trackduration", "mediaduration",
    "videoframerate", "avgbitrate", "imagewidth", "imageheight",
    "compressorid", "encodername",
    "make", "model", "lensmodel",
    "createtime", "creationdate", "modifydate",
    "gpslatitude", "gpslongitude"
}

IGNORE_TAGS = {
    "SourceFile",
    "FileName",
    "Directory",
    "FileSize",
    "FileModifyDate",
    "FileAccessDate",
    "FileInodeChangeDate",
    "FilePermissions",
    "ExifToolVersion"
}

KNOWN_CAMERAS = {
    "Apple",
    "Samsung",
    "Xiaomi",
    "Canon",
    "Sony",
    "Nikon",
    "Fuji",
    "Panasonic"
}