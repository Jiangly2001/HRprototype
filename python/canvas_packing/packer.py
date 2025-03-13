
class GrowingPacker:
    def __init__(self):
        self.root = None

    def fit(self, blocks):
        if not blocks:
            return

        # Initialize root node with the first block dimensions
        self.root = {"x": 0, "y": 0, "w": blocks[0]["w"], "h": blocks[0]["h"]}
        for block in blocks:
            node = self.find_node(self.root, block["w"], block["h"])
            if node:
                block["fit"] = self.split_node(node, block["w"], block["h"])
            else:
                block["fit"] = self.grow_node(block["w"], block["h"])
                if block["fit"] is None:
                    print(f"Warning: Block {block} could not be placed.")  # Log unplaced block

    def find_node(self, root, w, h):
        if "used" in root:
            return self.find_node(root.get("right"), w, h) or self.find_node(root.get("down"), w, h)
        elif w <= root["w"] and h <= root["h"]:
            return root
        return None

    def split_node(self, node, w, h):
        node["used"] = True
        node["down"] = {"x": node["x"], "y": node["y"] + h, "w": node["w"], "h": node["h"] - h}
        node["right"] = {"x": node["x"] + w, "y": node["y"], "w": node["w"] - w, "h": h}
        return node

    def grow_node(self, w, h):
        can_grow_down = w <= self.root["w"]
        can_grow_right = h <= self.root["h"]

        should_grow_right = can_grow_right and (self.root["h"] >= self.root["w"] + w)
        should_grow_down = can_grow_down and (self.root["w"] >= self.root["h"] + h)

        if should_grow_right:
            return self.grow_right(w, h)
        elif should_grow_down:
            return self.grow_down(w, h)
        elif can_grow_right:
            return self.grow_right(w, h)
        elif can_grow_down:
            return self.grow_down(w, h)
        return None

    def grow_right(self, w, h):
        self.root = {
            "used": True,
            "x": 0,
            "y": 0,
            "w": self.root["w"] + w,
            "h": self.root["h"],
            "down": self.root,
            "right": {"x": self.root["w"], "y": 0, "w": w, "h": self.root["h"]}
        }
        node = self.find_node(self.root, w, h)
        return self.split_node(node, w, h) if node else None

    def grow_down(self, w, h):
        self.root = {
            "used": True,
            "x": 0,
            "y": 0,
            "w": self.root["w"],
            "h": self.root["h"] + h,
            "down": {"x": 0, "y": self.root["h"], "w": self.root["w"], "h": h},
            "right": self.root
        }
        node = self.find_node(self.root, w, h)
        return self.split_node(node, w, h) if node else None

def sort_key(block):
    width = block["w"]
    color_sum = block["color"].sum()
    if width == 60:
        return (width, -color_sum)
    elif width == 30:
        return (width, color_sum)
    else:
        return (width, -color_sum)
