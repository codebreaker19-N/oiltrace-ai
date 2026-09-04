def get_raster_info(profile):
    """
    Extract basic raster metadata.
    """
    return {
        "width": profile.get("width"),
        "height": profile.get("height"),
        "count": profile.get("count"),
        "crs": str(profile.get("crs")),
        "transform": str(profile.get("transform")),
    }