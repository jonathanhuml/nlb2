"""Internal filtering cores adapted from the CASSM implementation."""

from __future__ import annotations

import math

import gpytorch
from linear_operator import operators
from linear_operator.operators import (
    AddedDiagLinearOperator,
    DiagLinearOperator,
    IdentityLinearOperator,
    KroneckerProductLinearOperator,
)
from linear_operator.operators._linear_operator import LinearOperator
import torch
from torch import Tensor, nn


class CASSMElboLoss(nn.Module):
    """Per-step CA-SSM evidence lower bound objective."""

    def forward(
        self,
        posterior_residual: Tensor,
        posterior_state_covariance,
        obs_noise: Tensor,
        cakf_mean_message: Tensor,
        mean_update_term: Tensor,
        projected_noise: Tensor,
        innovation_cholesky: Tensor,
        use_dense_projection: bool,
    ) -> Tensor:
        pos_diag = posterior_state_covariance.diagonal(dim1=-1, dim2=-2)[..., 0::2]
        inv_obs_noise = obs_noise.pow(-1)

        trace_cov = (inv_obs_noise * pos_diag).sum(-1)
        residual_quadratic = torch.mean(
            (posterior_residual.mT * inv_obs_noise) @ posterior_residual
        )
        normalizer = obs_noise.numel() * torch.log(2 * obs_noise.new_tensor(math.pi))

        expectation_term = 0.5 * (
            obs_noise.log().sum()
            + residual_quadratic
            + torch.mean(trace_cov)
            + normalizer
        )

        projected_noise_matrix = (
            projected_noise if use_dense_projection else torch.diag_embed(projected_noise)
        )
        cov_scaled_innovation = torch.cholesky_solve(
            projected_noise_matrix,
            innovation_cholesky,
            upper=False,
        )

        trace_term = cov_scaled_innovation.diagonal(dim1=-2, dim2=-1).sum(-1)
        projected_dim = projected_noise_matrix.shape[-1]
        noise_cholesky = torch.linalg.cholesky(projected_noise_matrix)
        logdet_ratio = (
            2.0 * noise_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            - 2.0 * innovation_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        kl_term = 0.5 * (
            torch.mean(cakf_mean_message.mT @ mean_update_term)
            + torch.mean(trace_term)
            - projected_dim
            - torch.mean(logdet_ratio)
        )

        return kl_term + expectation_term


class BlockDiagonalSparseLinearOperator(LinearOperator):
    """Sparse projection operator used by the computation-aware filter."""

    def __init__(
        self,
        non_zero_idcs: Tensor,
        blocks: Tensor,
        size_input_dim: int,
    ) -> None:
        super().__init__(non_zero_idcs, blocks, size_input_dim=size_input_dim)
        self.non_zero_idcs = torch.atleast_2d(non_zero_idcs)
        self.non_zero_idcs.requires_grad = False
        self.blocks = torch.atleast_2d(blocks)
        self.size_input_dim = size_input_dim

    def _matmul(self, rhs):
        if isinstance(rhs, AddedDiagLinearOperator):
            return self._matmul(rhs._linear_op) + self._matmul(rhs._diag_tensor)

        if isinstance(rhs, DiagLinearOperator):
            return BlockDiagonalSparseLinearOperator(
                non_zero_idcs=self.non_zero_idcs,
                blocks=rhs.diag()[self.non_zero_idcs] * self.blocks,
                size_input_dim=self.size_input_dim,
            ).to_dense()

        rhs_non_zero = rhs[..., self.non_zero_idcs, :]

        if rhs.ndim == 2 and rhs.shape[-1] == 1:
            return (self.blocks.unsqueeze(-1) * rhs_non_zero).sum(dim=-2)

        return (self.blocks.unsqueeze(-2) @ rhs_non_zero).squeeze(-2)

    def _size(self) -> torch.Size:
        return torch.Size((self.non_zero_idcs.shape[0], self.size_input_dim))

    def to_dense(self) -> Tensor:
        if self.size() == self.blocks.shape:
            return self.blocks
        return torch.zeros(
            (self.blocks.shape[0], self.size_input_dim),
            dtype=self.blocks.dtype,
            device=self.blocks.device,
        ).scatter_(src=self.blocks, index=self.non_zero_idcs, dim=1)


def _matern32_time_process_cov(
    delta_t: Tensor,
    sigma_f2: Tensor,
    ell: Tensor,
) -> Tensor:
    lam = delta_t.new_tensor(3.0).sqrt() / ell
    rho2 = torch.exp(-2.0 * lam * delta_t)
    u = lam * delta_t

    q11 = sigma_f2 * (1.0 - rho2 * (1.0 + 2.0 * u + 2.0 * u**2))
    q22 = sigma_f2 * lam**2 * (1.0 - rho2 * (1.0 - 2.0 * u + 2.0 * u**2))
    q12 = 2.0 * sigma_f2 * lam**3 * delta_t**2 * rho2

    return torch.stack(
        [torch.stack([q11, q12], -1), torch.stack([q12, q22], -1)],
        dim=-2,
    )


def _matern32_transition_matrix(delta_t: Tensor, ell: Tensor) -> Tensor:
    lam = delta_t.new_tensor(3.0).sqrt() / ell
    zero = torch.zeros_like(lam)
    one = torch.ones_like(lam)
    f_time = torch.stack(
        [torch.stack([zero, one]), torch.stack([-(lam**2), -2.0 * lam])],
    ).squeeze(-1)
    return torch.matrix_exp(f_time * delta_t)


def _matern32_time_stationary_cov(sigma_f2: Tensor, ell: Tensor) -> Tensor:
    lam = sigma_f2.new_tensor(3.0).sqrt() / ell
    return torch.stack(
        [
            torch.stack([sigma_f2, torch.zeros_like(sigma_f2)], -1),
            torch.stack([torch.zeros_like(sigma_f2), lam**2 * sigma_f2], -1),
        ],
        -2,
    )


def _log_marginal_likelihood(residual: Tensor, y_cholesky: Tensor) -> Tensor:
    n_neurons = residual.shape[1]
    loss1 = torch.mean(
        0.5
        * residual.transpose(1, 2)
        @ torch.cholesky_solve(input=residual, input2=y_cholesky, upper=False)
    )
    cholesky_diags = torch.diagonal(y_cholesky, offset=0, dim1=1, dim2=2)
    loss2 = torch.mean(torch.sum(torch.log(cholesky_diags), dim=1))
    loss3 = 0.5 * torch.log(2 * residual.new_tensor(math.pi)) * n_neurons
    return loss1 + loss2 + loss3


class ComputationAwareFilterSmoother(nn.Module):
    """Sparse computation-aware state-space filter used by NLB2 CASSM."""

    def __init__(
        self,
        projection_dim: int,
        nneurons: int,
        timesteps: int,
        device: torch.device,
        dt: float = 1.0,
        dataset_name: str | None = None,
        spatial_prior: Tensor | None = None,
        save_model: bool = False,
        use_dense_projection: bool = False,
    ) -> None:
        super().__init__()

        self.dim = nneurons
        self.projection_dim = projection_dim
        if not 1 <= self.projection_dim <= self.dim:
            raise ValueError("projection_dim must be positive and <= nneurons.")
        if self.dim % self.projection_dim != 0 and not use_dense_projection:
            raise ValueError(
                "Sparse CASSM requires nneurons to be divisible by projection_dim. "
                "Choose a divisor or set use_dense_projection=True."
            )

        self.state_dim = 2 * self.dim
        self.t = timesteps
        self.device = device
        self.save_model = save_model
        self.register_buffer("dt", torch.tensor(float(dt), device=device))

        self.raw_sigma_f = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.raw_ell = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.softplus = nn.Softplus()
        self.loss_fn = CASSMElboLoss()

        if spatial_prior is not None:
            self.register_buffer("latent_locations", spatial_prior.float().to(device))
        else:
            self.latent_locations = nn.Parameter(
                torch.randn(self.dim, 3, device=device)
                / torch.tensor(float(self.dim), device=device).sqrt()
            )

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=3)
        )
        self.spatial_kernel(self.latent_locations)

        self.use_dense_projection = use_dense_projection
        self.obs_noise_values = nn.Parameter(1e-2 * torch.ones(self.dim, device=device))
        self.belief_initial_state = nn.Parameter(
            torch.zeros(self.state_dim, 1, device=device)
        )

        if self.use_dense_projection:
            self.dense_projection = nn.Parameter(
                torch.empty(self.projection_dim, self.dim, device=device)
            )
            nn.init.orthogonal_(self.dense_projection)
        else:
            self.projection = nn.Parameter(
                torch.ones(
                    (self.projection_dim, self.dim // self.projection_dim),
                    device=device,
                )
            )
            self.register_buffer(
                "projection_indices",
                torch.arange(self.dim, device=device).reshape(
                    self.projection_dim,
                    self.dim // self.projection_dim,
                ),
            )

        self.observation_matrix = KroneckerProductLinearOperator(
            IdentityLinearOperator(
                self.dim,
                device=device,
                dtype=self.obs_noise_values.dtype,
            ),
            torch.tensor(
                [[1.0, 0.0]],
                device=device,
                dtype=self.obs_noise_values.dtype,
            ),
        )

        if self.save_model:
            raise ValueError(
                "Model-local save_model is unsupported; use Experiment output_dir. "
                "Experiment.run saves the fitted checkpoint."
            )

    def _build_dynamics(self):
        ell = self.softplus(self.raw_ell)
        sigma_f2 = self.softplus(self.raw_sigma_f)

        a_t = _matern32_transition_matrix(self.dt, ell)
        transition_matrix = KroneckerProductLinearOperator(
            IdentityLinearOperator(self.dim, device=self.device, dtype=a_t.dtype),
            a_t,
        )

        sigma_inf_t = _matern32_time_stationary_cov(sigma_f2, ell)
        sigma_inf_op = KroneckerProductLinearOperator(
            self.spatial_kernel(self.latent_locations),
            sigma_inf_t,
        )

        return transition_matrix, sigma_inf_op

    def _build_projected_obs(self):
        if self.use_dense_projection:
            h_proj = (self.dense_projection @ self.observation_matrix).unsqueeze(0)
            r_proj = (
                self.dense_projection
                * self.softplus(self.obs_noise_values)
                @ self.dense_projection.mT
            )
        else:
            projection = BlockDiagonalSparseLinearOperator(
                non_zero_idcs=self.projection_indices,
                blocks=self.projection,
                size_input_dim=self.dim,
            )
            h_proj = (projection @ self.observation_matrix).to_dense().unsqueeze(0)
            r_proj = (
                self.projection.pow(2)
                * self.softplus(self.obs_noise_values)[self.projection_indices]
            ).sum(dim=1)

        return h_proj, r_proj

    def _truncate_downdate(self, matrix: Tensor) -> Tensor:
        with torch.no_grad():
            _, _, vh = torch.linalg.svd(matrix.detach(), full_matrices=False)
            truncation_basis = vh.mH[..., : self.projection_dim]
        return matrix @ truncation_basis

    def filter(self, data: Tensor, return_type: str = "forward"):
        num_trials, time_steps = data.shape[:2]

        if return_type == "prediction":
            updated_belief_state_means = torch.empty(
                (num_trials, time_steps, self.state_dim),
                device=self.device,
                dtype=data.dtype,
            )
            updated_belief_obs_vars = torch.empty(
                (num_trials, time_steps, self.dim),
                device=self.device,
                dtype=data.dtype,
            )

        prior_belief_state_mean = self.belief_initial_state.unsqueeze(0).expand(
            num_trials,
            -1,
            -1,
        )

        downdate_sqrt = torch.zeros(
            size=(1, self.state_dim, self.projection_dim),
            device=self.device,
            dtype=data.dtype,
        )
        loss = data.new_zeros(())

        transition_matrix, sigma_inf_op = self._build_dynamics()
        h_proj, r_proj = self._build_projected_obs()

        prior_belief_state_cov_op = sigma_inf_op - operators.RootLinearOperator(
            downdate_sqrt
        )

        for t in range(time_steps):
            tmp = prior_belief_state_cov_op.matmul(h_proj.mT)
            innovation_matrix = h_proj @ tmp
            if self.use_dense_projection:
                innovation_matrix = innovation_matrix + r_proj.unsqueeze(0)
            else:
                innovation_matrix.diagonal(dim1=-2, dim2=-1).add_(r_proj)

            prior_predictive_residual = (
                data[:, t, :].unsqueeze(-1)
                - self.observation_matrix @ prior_belief_state_mean
            )

            if self.use_dense_projection:
                projected_residual = self.dense_projection @ prior_predictive_residual
            else:
                projected_residual = (
                    (
                        self.projection
                        * prior_predictive_residual.squeeze(-1)[
                            ...,
                            self.projection_indices,
                        ]
                    )
                    .sum(-1)
                    .unsqueeze(-1)
                )

            cholesky = torch.linalg.cholesky(innovation_matrix, upper=False)

            cakf_mean_message = h_proj.mT @ torch.cholesky_solve(
                projected_residual,
                cholesky,
                upper=False,
            )
            cakf_cov_message = torch.linalg.solve_triangular(
                cholesky,
                h_proj,
                upper=False,
            ).mT

            mean_update_term = prior_belief_state_cov_op.matmul(cakf_mean_message)
            updated_belief_state_mean = prior_belief_state_mean + mean_update_term
            scaled_cov = prior_belief_state_cov_op.matmul(cakf_cov_message)
            updated_belief_state_cov_op = (
                prior_belief_state_cov_op - operators.RootLinearOperator(scaled_cov)
            )

            loss = loss + self.loss_fn(
                posterior_residual=(
                    prior_predictive_residual - self.observation_matrix @ mean_update_term
                ),
                posterior_state_covariance=updated_belief_state_cov_op,
                obs_noise=self.softplus(self.obs_noise_values),
                cakf_mean_message=cakf_mean_message,
                mean_update_term=mean_update_term,
                projected_noise=r_proj,
                innovation_cholesky=cholesky,
                use_dense_projection=self.use_dense_projection,
            )

            if return_type == "prediction":
                pos_diag = updated_belief_state_cov_op.diagonal(dim1=-1, dim2=-2)[..., 0::2]
                updated_belief_obs_vars[:, t, :] = pos_diag + self.softplus(
                    self.obs_noise_values
                )
                updated_belief_state_means[:, t, :] = updated_belief_state_mean[:, :, 0]

            # Evaluate the current posterior before compressing and propagating
            # its covariance representation to the next time step.
            m_trunc = self._truncate_downdate(torch.cat([downdate_sqrt, scaled_cov], dim=-1))
            prior_belief_state_mean = transition_matrix @ updated_belief_state_mean
            downdate_sqrt = transition_matrix @ m_trunc
            prior_belief_state_cov_op = (
                sigma_inf_op - operators.RootLinearOperator(downdate_sqrt)
            )

        loss = loss * (1 / time_steps) * (1 / self.dim)

        if return_type == "forward":
            return loss
        if return_type == "prediction":
            return updated_belief_state_means, updated_belief_obs_vars
        raise ValueError(
            f"Unknown return_type '{return_type}'. Expected: forward | prediction"
        )

    def forward(self, data: Tensor) -> Tensor:
        return self.filter(data, return_type="forward")


class DenseKalmanFilterSmoother(nn.Module):
    """Dense Kalman filter baseline used by NLB2 Kalman."""

    def __init__(
        self,
        nneurons: int,
        timesteps: int,
        device: torch.device,
        dt: float = 1.0,
        dataset_name: str | None = None,
        save_model: bool = False,
    ) -> None:
        super().__init__()

        self.dim = nneurons
        self.latent_dim = self.dim
        self.state_dim = 2 * self.dim
        self.t = timesteps
        self.device = device
        self.save_model = save_model
        self.register_buffer("dt", torch.tensor(float(dt), device=device))

        self.raw_sigma_f = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.raw_ell = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.softplus = nn.Softplus()

        self.latent_locations = nn.Parameter(
            torch.arange(self.dim, device=device).float().unsqueeze(-1)
        )
        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        self.obs_noise_values = nn.Parameter(1e-1 * torch.ones(self.dim, device=device))

        if self.save_model:
            raise ValueError(
                "Model-local save_model is unsupported; use Experiment output_dir. "
                "Experiment.run saves the fitted checkpoint."
            )

    def build_matern_observation_matrix(self) -> Tensor:
        eye = torch.eye(
            self.dim,
            device=self.device,
            dtype=self.obs_noise_values.dtype,
        )
        h_time = torch.tensor(
            [[1.0, 0.0]],
            device=self.device,
            dtype=self.obs_noise_values.dtype,
        )
        return torch.kron(eye, h_time)

    def spatial_cov(self) -> Tensor:
        return self.spatial_kernel(self.latent_locations).to_dense()

    def filter(
        self,
        data: Tensor,
        return_type: str = "for_prediction",
        holdout: bool = False,
    ):
        num_trials, time_steps = data.shape[:2]

        updated_belief_state_means = torch.zeros(
            size=(num_trials, time_steps, self.state_dim),
            device=self.device,
            dtype=data.dtype,
        )
        updated_belief_state_covs = torch.zeros(
            size=(1, time_steps, self.state_dim, self.state_dim),
            device=self.device,
            dtype=data.dtype,
        )

        prior_belief_state_mean = torch.zeros(
            size=(num_trials, self.state_dim, 1),
            device=self.device,
            dtype=data.dtype,
        )
        prior_belief_state_cov = IdentityLinearOperator(
            self.state_dim,
            batch_shape=(1,),
            device=self.device,
            dtype=data.dtype,
        )
        loss = data.new_zeros(())

        ell = self.softplus(self.raw_ell)
        sigma_f2 = self.softplus(self.raw_sigma_f)
        obs_noise = self.softplus(self.obs_noise_values)

        a_t = _matern32_transition_matrix(self.dt, ell)
        transition_matrix = torch.kron(
            torch.eye(self.latent_dim, device=self.device, dtype=a_t.dtype),
            a_t,
        ).unsqueeze(0)

        q_t = _matern32_time_process_cov(self.dt, sigma_f2, ell)
        process_noise = torch.kron(self.spatial_cov(), q_t)

        if holdout:
            n_held_in = data.shape[2]
            truncated = torch.eye(self.dim, device=self.device, dtype=data.dtype)[
                :n_held_in,
                :,
            ]
            h_time = torch.tensor(
                [[1.0, 0.0]],
                device=self.device,
                dtype=data.dtype,
            )
            observation_matrix = torch.kron(truncated, h_time).unsqueeze(0)
            obs_noise = obs_noise[:n_held_in]
        else:
            observation_matrix = self.build_matern_observation_matrix().unsqueeze(0)

        for t in range(time_steps):
            innovation_matrix = (
                observation_matrix
                @ prior_belief_state_cov
                @ observation_matrix.mT
            )
            innovation_matrix.diagonal(dim1=-2, dim2=-1).add_(obs_noise)

            prior_predictive_residual = (
                data[:, t, :].unsqueeze(-1)
                - torch.matmul(observation_matrix, prior_belief_state_mean)
            )

            innovation_cholesky = torch.linalg.cholesky(innovation_matrix, upper=False)
            innovation_inverse_obs_matrix = torch.cholesky_solve(
                input=observation_matrix,
                input2=innovation_cholesky,
                upper=False,
            )
            kalman_gain = (
                prior_belief_state_cov
                @ innovation_inverse_obs_matrix.transpose(1, 2)
            )

            updated_belief_state_mean = prior_belief_state_mean + torch.matmul(
                kalman_gain,
                prior_predictive_residual,
            )
            joseph_gain = kalman_gain @ observation_matrix
            joseph_gain.diagonal(dim1=-2, dim2=-1).sub_(1.0)
            updated_belief_state_cov = (
                joseph_gain
                @ prior_belief_state_cov
                @ joseph_gain.transpose(1, 2)
                + kalman_gain * obs_noise @ kalman_gain.transpose(1, 2)
            )

            prior_belief_state_mean = torch.matmul(
                transition_matrix,
                updated_belief_state_mean,
            )
            prior_belief_state_cov = (
                transition_matrix
                @ updated_belief_state_cov
                @ transition_matrix.transpose(1, 2)
                + process_noise
            )

            loss = loss + _log_marginal_likelihood(
                prior_predictive_residual,
                innovation_cholesky,
            )

            updated_belief_state_means[:, t, :] = updated_belief_state_mean[:, :, 0]
            updated_belief_state_covs[:, t, :, :] = updated_belief_state_cov

        loss = loss / (time_steps * self.dim)

        if return_type == "for_forward":
            return loss
        return updated_belief_state_means, updated_belief_state_covs

    def forward(self, data: Tensor) -> Tensor:
        return self.filter(data, return_type="for_forward")
