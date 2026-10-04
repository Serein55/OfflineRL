"""OpenPI legacy LIBERO extra delta convention, before action normalization."""
def extra_delta(actions, state):
    result=actions.copy()
    result[..., :6]-=state[..., None, :6]
    return result


class ExtraDeltaInference:
    """Invert using the raw observation at chunk generation, never the later state."""
    extra_delta_transform=False

    def reset(self):
        super().reset()
        self._extra_delta_anchor=None

    def select_action(self,batch,noise=None):
        if self.extra_delta_transform and not self._action_queue:
            self._extra_delta_anchor=batch['observation.state'][..., :6].clone()
        action=super().select_action(batch,noise=noise)
        if self.extra_delta_transform:
            action=action.clone()
            action[..., :6]+=self._extra_delta_anchor
        return action
