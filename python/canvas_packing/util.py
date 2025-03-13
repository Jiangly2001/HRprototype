def sort_key(block):
    width = block["w"]
    color_sum = block["color"].sum()
    if width == 60:
        return (width, -color_sum)
    elif width == 30:
        return (width, color_sum)
    else:
        return (width, -color_sum)