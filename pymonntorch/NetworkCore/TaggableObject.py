import torch


class TaggableObject(torch.nn.Module):
    """This is the base class for all taggable objects.

    This class is used to add tags to objects and to search for objects with a specific tag.
    It is a torch.nn.Module, so that any object class inheriting from this class can be used
    as a torch.nn.Module to benefit from its functionality, e.g. `state_dict`.

    Attributes:
        tags (list): List of tags.
        tag_shortcuts (dict): Cache for faster search.
        device (str): Device on which the object is located. The default is "cpu".
    """

    def __init__(self, tag, device="cpu"):
        """Initialize the object.

        Args:
            tag (str): Tag to add to the object. It can also be a comma-separated string of multiple tags.
            device (str): Device on which the object is located.
        """
        self.device = device
        super().__init__()

        self.requires_grad_(False)
        self.tags = []
        self.clear_cache()

        #: dict of matrices that are evaluated on the fly
        self._mat_eval_dict = {}

        if tag is not None:
            self.add_tag(tag)
        self.add_tag(self.__class__.__name__)

    @property
    def tag(self):
        return self.tags[0]

    def has_module(self, tag):
        """Check if object has a module with a specific tag.

        Args:
            tag (str): Tag to search for.

        Returns:
            TaggableObject: The object with the tag. If no object is found, None is returned.

        Note: The returned object can be any object inheriting from TaggableObject.
        """
        return self[tag, 0] is not None

    def find_objects(self, key):
        """Find objects with a specific tag.

        This method should be overridden for deeper search.

        Args:
            key (str): Tag to search for.

        Returns:
            list: List of objects with the tag.
        """
        result = []
        return result

    def clear_cache(self):
        """Clear the tag cache for faster search.

        This method is called automatically when a tag is added or removed.
        """
        self.tag_shortcuts = {}

    def set_tag_attrs(self, tag, attr, value):
        """Set an attribute of all objects with a specific tag.

        Args:
            tag (str): Tag to search for.
            attr (str): Attribute to set.
            value (any): Value to set to the attribute.
        """
        for obj in self[tag]:
            setattr(obj, attr, value)

    def call_tag_functions(self, tag, attr, **args):
        """Call a function of all objects with a specific tag.

        Args:
            tag (str): Tag to search for.
            attr (str): Function to call.
            args (dict): Arguments to pass to the function.
        """
        for obj in self[tag]:
            getattr(obj, attr)(args)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            k = key[0]
            index = key[1]
        else:
            k = key
            index = None

        # cache
        if key in self.tag_shortcuts:
            result = self.tag_shortcuts[key]

            if index is not None:
                if len(result) > 0:
                    return result[index]
                else:
                    return None
            else:
                return result

        # normal search
        result = []
        if k in self.tags or (k is type and isinstance(self, k)):
            result.append(self)

        result += self.find_objects(k)

        if k not in self.tag_shortcuts:
            self.tag_shortcuts[k] = result

        if index is not None:
            if len(result) > 0:
                result = result[index]
            else:
                result = None

        return result

    def add_tag(self, tag):
        """Add a tag to the object.

        Args:
            tag (str): Tag to add.

        Returns:
            TaggableObject: The object itself.
        """
        for subtag in tag.split(","):
            self.tags.append(subtag.strip())
        return self
