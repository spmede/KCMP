from collections import OrderedDict
import hashlib

class ImageCache:
    def __init__(self, max_size=128):
        self.cache = OrderedDict()
        self.max_size = max_size

    def _hash_image(self, img):
        if isinstance(img, str):
            return hashlib.md5(img.encode()).hexdigest()
        return hashlib.md5(img.tobytes()).hexdigest()

    def get(self, img):
        key = self._hash_image(img)
        return self.cache.get(key)

    def set(self, img, value):
        key = self._hash_image(img)
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value
