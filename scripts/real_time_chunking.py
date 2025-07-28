import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Literal, TypeAlias

PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


def realtime_action(
    rng: jax.Array,
    obs: jax.Array,
    num_steps: int,
    prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
    inference_delay: int,
    prefix_attention_horizon: int,
    prefix_attention_schedule: PrefixAttentionSchedule,
    max_guidance_weight: float,
    ) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch
            def pinv_corrected_velocity(obs, x_t, y, t):
                def denoiser(x_t):
                    v_t = self(obs[None], x_t[None], t)[0]
                    return x_t + v_t * (1 - t), v_t

                x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay, prefix_attention_horizon, self.action_chunk_size, prefix_attention_schedule
                )
                error = (y - x_1) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]
                # constants from paper
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
                c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                return v_t + guidance_weight * pinv_correction

            v_t = pinv_corrected_velocity(obs, x_t, prev_action_chunk, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1
