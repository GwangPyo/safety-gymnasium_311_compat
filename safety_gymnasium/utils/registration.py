# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Register and make environments."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from gymnasium import Env, error, logger
from gymnasium.envs.registration import namespace  # noqa: F401 # pylint: disable=unused-import
from gymnasium.envs.registration import spec  # noqa: F401 # pylint: disable=unused-import
from gymnasium.envs.registration import EnvSpec, _check_metadata, _find_spec, load_env_creator
from gymnasium.envs.registration import register as gymnasium_register
from gymnasium.wrappers import HumanRendering, OrderEnforcing, RenderCollection

from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafePassiveEnvChecker, SafeTimeLimit
"""A compatibility wrapper converting an old-style environment into a valid environment."""
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import gymnasium as gym
from gymnasium import logger
from gymnasium.core import ObsType
from gymnasium.utils.step_api_compatibility import (
    convert_to_terminated_truncated_step_api,
)


@runtime_checkable
class LegacyEnv(Protocol):
    """A protocol for environments using the old step API."""

    observation_space: gym.Space
    action_space: gym.Space

    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""
        ...

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Run one timestep of the environment's dynamics."""
        ...

    def render(self, mode: Optional[str] = "human") -> Any:
        """Render the environment."""
        ...

    def close(self):
        """Close the environment."""
        ...

    def seed(self, seed: Optional[int] = None):
        """Set the seed for this env's random number generator(s)."""
        ...


class EnvCompatibility(gym.Env):
    r"""A wrapper which can transform an environment from the old API to the new API.

    Old step API refers to step() method returning (observation, reward, done, info), and reset() only retuning the observation.
    New step API refers to step() method returning (observation, reward, terminated, truncated, info) and reset() returning (observation, info).
    (Refer to docs for details on the API change)

    Known limitations:
    - Environments that use `self.np_random` might not work as expected.
    """

    def __init__(self, old_env: LegacyEnv, render_mode: Optional[str] = None):
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.

        Args:
            old_env (LegacyEnv): the env to wrap, implemented with the old API
            render_mode (str): the render mode to use when rendering the environment, passed automatically to env.render
        """
        logger.deprecation(
            "The `gymnasium.make(..., apply_api_compatibility=...)` parameter is deprecated and will be removed in v1.0. "
            "Instead use `gymnasium.make('GymV21Environment-v0', env_name=...)` or `from shimmy import GymV21CompatibilityV0`"
        )

        self.env = old_env
        self.metadata = getattr(old_env, "metadata", {"render_modes": []})
        self.render_mode = render_mode
        self.reward_range = getattr(old_env, "reward_range", None)
        self.spec = getattr(old_env, "spec", None)

        self.observation_space = old_env.observation_space
        self.action_space = old_env.action_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        if seed is not None:
            self.env.seed(seed)
        # Options are ignored

        if self.render_mode == "human":
            self.render()

        return self.env.reset(), {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.env.step(action)

        if self.render_mode == "human":
            self.render()

        return convert_to_terminated_truncated_step_api((obs, reward, done, info))

    def render(self) -> Any:
        """Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.env.render(mode=self.render_mode)

    def close(self):
        """Closes the environment."""
        self.env.close()

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

safe_registry = set()


def register(**kwargs):
    """Register an environment."""
    safe_registry.add(kwargs['id'])
    gymnasium_register(**kwargs)


# pylint: disable-next=too-many-arguments,too-many-branches,too-many-statements,too-many-locals
def make(
    id: str | EnvSpec,  # pylint: disable=invalid-name,redefined-builtin
    max_episode_steps: int | None = None,
    autoreset: bool | None = None,
    apply_api_compatibility: bool | None = None,
    disable_env_checker: bool | None = None,
    **kwargs: Any,
) -> Env:
    """Creates an environment previously registered with :meth:`gymnasium.register` or a :class:`EnvSpec`.

    To find all available environments use ``gymnasium.envs.registry.keys()`` for all valid ids.

    Args:
        id: A string for the environment id or a :class:`EnvSpec`. Optionally if using a string,
            a module to import can be included, e.g. ``'module:Env-v0'``.
            This is equivalent to importing the module first to register the environment
            followed by making the environment.
        max_episode_steps: Maximum length of an episode, can override the registered
            :class:`EnvSpec` ``max_episode_steps``.
            The value is used by :class:`gymnasium.wrappers.TimeLimit`.
        autoreset: Whether to automatically reset the environment after each episode
        (:class:`gymnasium.wrappers.AutoResetWrapper`).
        apply_api_compatibility: Whether to wrap the environment with the
            :class:`gymnasium.wrappers.StepAPICompatibility` wrapper that
            converts the environment step from a done bool to return termination and truncation bools.
            By default, the argument is None in which the :class:`EnvSpec` ``apply_api_compatibility`` is used,
            otherwise this variable is used in favor.
        disable_env_checker: If to add :class:`gymnasium.wrappers.PassiveEnvChecker`, ``None`` will default to the
            :class:`EnvSpec` ``disable_env_checker`` value otherwise use this value will be used.
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment with wrappers applied.

    Raises:
        Error: If the ``id`` doesn't exist in the :attr:`registry`
    """
    if isinstance(id, EnvSpec):
        env_spec = id
        if not hasattr(env_spec, 'additional_wrappers'):
            logger.warn(
                'The env spec passed to `make` does not have a `additional_wrappers`,'
                'set it to an empty tuple. Env_spec={env_spec}',
            )
            env_spec.additional_wrappers = ()
    else:
        # For string id's, load the environment spec from the registry then make the environment spec
        assert isinstance(id, str)
        assert id in safe_registry, f'Environment {id} is not registered in safety-gymnasium.'

        # The environment name can include an unloaded module in "module:env_name" style
        env_spec = _find_spec(id)

    assert isinstance(env_spec, EnvSpec)

    # Update the env spec kwargs with the `make` kwargs
    env_spec_kwargs = copy.deepcopy(env_spec.kwargs)
    # one level deeper update of `env_spec_kwargs`
    for key, value in kwargs.items():
        if key in env_spec_kwargs and isinstance(value, Mapping):
            # updating not copying dictionaries
            if key == 'config':
                # Do not update 'agent_name' or 'task_name'
                if 'agent_name' in value and 'agent_name' in env_spec_kwargs[key]:
                    assert (
                        env_spec_kwargs[key]['agent_name'] == value['agent_name']
                    ), "Agent specified in the config doesn't match the agent in the `env_id`"
                if 'task_name' in value and 'task_name' in env_spec_kwargs[key]:
                    assert (
                        env_spec_kwargs[key]['task_name'] == value['task_name']
                    ), "Task specified in the config doesn't match the task in the `env_id`"
            env_spec_kwargs[key].update(value)
        else:
            env_spec_kwargs[key] = value

    # Load the environment creator
    if env_spec.entry_point is None:
        raise error.Error(f'{env_spec.id} registered but entry_point is not specified')
    if callable(env_spec.entry_point):
        env_creator = env_spec.entry_point
    else:
        # Assume it's a string
        env_creator = load_env_creator(env_spec.entry_point)

    # Determine if to use the rendering
    render_modes: list[str] | None = None
    if hasattr(env_creator, 'metadata'):
        _check_metadata(env_creator.metadata)
        render_modes = env_creator.metadata.get('render_modes')
    render_mode = env_spec_kwargs.get('render_mode')
    apply_human_rendering = False
    apply_render_collection = False

    # If mode is not valid, try applying HumanRendering/RenderCollection wrappers
    if render_mode is not None and render_modes is not None and render_mode not in render_modes:
        displayable_modes = {'rgb_array', 'rgb_array_list'}.intersection(render_modes)
        if render_mode == 'human' and len(displayable_modes) > 0:
            logger.warn(
                "You are trying to use 'human' rendering for an environment that doesn't natively support it. "
                'The HumanRendering wrapper is being applied to your environment.',
            )
            env_spec_kwargs['render_mode'] = displayable_modes.pop()
            apply_human_rendering = True
        elif render_mode.endswith('_list') and render_mode[: -len('_list')] in render_modes:
            env_spec_kwargs['render_mode'] = render_mode[: -len('_list')]
            apply_render_collection = True
        else:
            logger.warn(
                f'The environment is being initialised with render_mode={render_mode!r} '
                f'that is not in the possible render_modes ({render_modes}).',
            )

    if apply_api_compatibility:
        # If we use the compatibility layer, we treat the render mode explicitly and don't pass it to the env creator
        render_mode = env_spec_kwargs.pop('render_mode', None)
    else:
        render_mode = None

    try:
        env = env_creator(**env_spec_kwargs)
    except TypeError as e:
        if (
            str(e).find("got an unexpected keyword argument 'render_mode'") >= 0
            and apply_human_rendering
        ):
            raise error.Error(
                f"You passed render_mode='human' although {env_spec.id} doesn't implement human-rendering natively. "
                'Gym tried to apply the HumanRendering wrapper but it looks like your environment is using the old '
                'rendering API, which is not supported by the HumanRendering wrapper.',
            ) from e
        raise e

    # Set the minimal env spec for the environment.
    env.unwrapped.spec = EnvSpec(
        id=env_spec.id,
        entry_point=env_spec.entry_point,
        reward_threshold=env_spec.reward_threshold,
        nondeterministic=env_spec.nondeterministic,
        max_episode_steps=None,
        order_enforce=False,
        disable_env_checker=True,
        kwargs=env_spec_kwargs,
        additional_wrappers=(),
        vector_entry_point=env_spec.vector_entry_point,
    )

    # Check if pre-wrapped wrappers
    assert env.spec is not None
    num_prior_wrappers = len(env.spec.additional_wrappers)
    if env_spec.additional_wrappers[:num_prior_wrappers] != env.spec.additional_wrappers:
        for env_spec_wrapper_spec, recreated_wrapper_spec in zip(
            env_spec.additional_wrappers,
            env.spec.additional_wrappers,
        ):
            raise ValueError(
                f"The environment's wrapper spec {recreated_wrapper_spec} is different from"
                f'the saved `EnvSpec` additional wrapper {env_spec_wrapper_spec}',
            )

    # Add step API wrapper
    if apply_api_compatibility is True:
        env = EnvCompatibility(env, render_mode)

    # Run the environment checker as the lowest level wrapper
    if disable_env_checker is False or (
        disable_env_checker is None and env_spec.disable_env_checker is False
    ):
        env = SafePassiveEnvChecker(env)

    # Add the order enforcing wrapper
    if env_spec.order_enforce:
        env = OrderEnforcing(env)

    # Add the time limit wrapper
    if max_episode_steps is not None:
        env = SafeTimeLimit(env, max_episode_steps)
    elif env_spec.max_episode_steps is not None:
        env = SafeTimeLimit(env, env_spec.max_episode_steps)

    # Add the auto-reset wrapper
    if autoreset is True:
        env = SafeAutoResetWrapper(env)

    for wrapper_spec in env_spec.additional_wrappers[num_prior_wrappers:]:
        if wrapper_spec.kwargs is None:
            raise ValueError(
                f'{wrapper_spec.name} wrapper does not inherit from'
                '`gymnasium.utils.RecordConstructorArgs`, therefore, the wrapper cannot be recreated.',
            )

        env = load_env_creator(wrapper_spec.entry_point)(env=env, **wrapper_spec.kwargs)

    # Add human rendering wrapper
    if apply_human_rendering:
        env = HumanRendering(env)
    elif apply_render_collection:
        env = RenderCollection(env)

    return env
