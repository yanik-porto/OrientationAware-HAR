class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        # assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                print("not implemented")
                exit()
                # transform = build_from_cfg(transform, PIPELINES)
                # self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
    def update(self):
        for t in self.transforms:
            if hasattr(t.__class__, 'update') and callable(getattr(t.__class__, 'update')):
                t.update()
                print(t.__class__, " updated")

    def update_with_losses(self, losses):
        for t in self.transforms:
            if hasattr(t.__class__, 'update_with_losses') and callable(getattr(t.__class__, 'update_with_losses')):
                t.update_with_losses(losses)
                print(t.__class__, " updated with losses")