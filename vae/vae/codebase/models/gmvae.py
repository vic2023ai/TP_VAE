# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mezcla de prior Gaussianos
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Ponderación uniforme
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Calcula el Límite Inferior de Evidencia (Evidence Lower Bound), KL y costos de Reconstrucción

        Args:
            x: tensor: (batch, dim): Observaciones

        Returns:
            nelbo: tensor: (): Límite inferior de evidencia negativo
            kl: tensor: (): Divergencia KL del ELBO al prior
            rec: tensor: (): Término de reconstrucción del ELBO
        """
        ################################################################################
        # TODO: Modificar/completar el código aquí
        # Calcular el Límite Inferior de Evidencia negativo y su descomposición en KL y Rec
        #
        # Para ayudarte a empezar, hemos calculado la mezcla de prior Gaussianos
        # prior = (m_mixture, v_mixture) para ti, donde
        # m_mixture y v_mixture cada uno tiene forma (1, self.k, self.z_dim)
        #
        # Nota que nelbo = kl + rec
        #
        # Las salidas deben ser todas escalares
        ################################################################################
        # prior aprendible
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        ################################################################################
        # Fin de modificación del código
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Calcula el Límite del Autoencoder con Ponderación por Importancia
        Adicionalmente, también calculamos los términos KL y de reconstrucción del ELBO

        Args:
            x: tensor: (batch, dim): Observaciones
            iw: int: (): Número de muestras ponderadas por importancia

        Returns:
            niwae: tensor: (): Límite IWAE negativo
            kl: tensor: (): Divergencia KL del ELBO al prior
            rec: tensor: (): Término de reconstrucción del ELBO
        """
        ################################################################################
        # TODO: Modificar/completar el código aquí
        # Calcular niwae (IWAE negativo) con iw muestras de importancia, y la descomposición
        # KL y Rec del Límite Inferior de Evidencia
        #
        # Las salidas deben ser todas escalares
        ################################################################################

        ################################################################################
        # Fin de modificación del código
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
