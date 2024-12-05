#import "@preview/charged-ieee:0.1.2": ieee

#show: ieee.with(
  title: [Diffusion Soup: A Recipe for Improved Diffusion Models],
  abstract: [
    Recent advances in generative machine learning models have made it easier than ever to access
    powerful image manipulation tools such as text-to-image generation, generative upscaling, and
    damaged image inpainting. These tools are powered by diffusion models which generate novel
    outputs through a process of sequentially "denoising" inputs. In this paper, we demonstrate that
    it is possible to improve latent diffusion models by applying the Model Soups approach: training
    a large ensemble of models and combining their parameters. Our proof-of-concept uses cut-down
    LDMs and is trained using the ImageNet dataset.
  ],
  authors: (
    (
      name: "Marcus Mellor",
      department: [VLSI Computer Architecture Research Laboratory],
      organization: [College of Engineering, Architecture, and Technology],
      location: [Stillwater, OK],
      email: "marcus.mellor@okstate.edu"
    ),
    (
      name: "Jacob Pease",
      department: [VLSI Computer Architecture Research Laboratory],
      organization: [College of Engineering, Architecture, and Technology],
      location: [Stillwater, OK],
      email: "jacob.pease@okstate.edu"
    ),
  ),
  index-terms: ("machine learning", "peer review"),
  bibliography: bibliography("refs.bib"),
)

= Introduction



== Paper Overview

= Background

== Latent Diffusion
Latent Diffusion, as described in @ldm, differs from prior state-of-the-art diffusion models by
performing denoising operations in latent space rather than pixel space. This significantly reduces
the computational resources required for training when compared to diffusion models which operate
on pixel space representations. At the time of publication, latent diffusion models (LDMs) offered
state-of-the-art image inpainting and provided competitive capabilities for unconditional image
generation, generative upscaling, and scene synthesis from input images with labeled regions. In
this paper we test only the image generation capabilities of LDMs.

Like established diffusion architectures, the key operation of LDMs is the diffusion process. In
this process, outputs are generated from conditioned noise through many sequential denoising
steps. However, LDMs make two key structural changes to the traditional diffusion formula. First,
in order to operate on latent representations, LDMs introduce pretrained autoencoder networks to
capture key parameters of training data. Second, LDMs introduce cross-attention layers which enable
generative sampling from the learned latent space.

== Model Soups
As described in @model-soups, combining parameters from an ensemble of finely-tuned models can
produce better results than any of the individual models. The authors present two approaches:
a "Uniform Soup" recipe that simply averages the parameters of all models, and a "Greedy Soup"
recipe which uses a greedy algorithm to include only those models which improve the recipe's
performance against a test dataset. The authors of @model-soups apply this to transformer-
based models. In this paper, we instead evaluate their approach with LDMs.

Using their codebase, we duplicated their experiment as shown in @fig-soup-results. Computations
were performed on an AMD Radeon 6900 XT.

#figure(
    image("soups.png"),
    caption: [
        Results from replicating the experiment posed in @model-soups.
    ]
) <fig-soup-results>

Note that our results vary slightly from the original experiment. We suspect, but cannot confirm,
that this is due to differences in the low-level hardware and software used to compute the results.

= Methods
Due to limited computational resources, we used miniaturized latent diffusion models ("mini-LDMs")
with a fraction of the usual parameter count. These mini-LDMs are based on the `cin-ldm-vq-f8.yaml`
configuration provided in @ldm. Additionally, training is limited to a random selection of 10,000
images from the ImageNet dataset. Each model draws a new random selection of images from the
dataset.

There is one huge advantage to this approach: it reduces the necessary computational resources down
to a level that is within our grasp. However, it comes with several major disadvantages. Since each
model has to learn its own latent representation of the dataset, and each model is trained on a
new random subsample of the ImageNet dataset, no single model will learn a well-rounded latent
representation. This is demonstrated in our unconditional image generation testing in INSERT
SECTION HERE. In addition, since we only train with a single pass through the data, our models are
likely to be poorly tuned. As such, we expect our results to demonstrate a proof of concept rather
than a major advancement in the state of the art.

All training was performed on a Radeon RX 6900 XT using the ROCm software stack for just-in-time
translation of CUDA code.

== Training Mini-LDM Models

== Mini-LDM Soup Recipes

=== Uniform Soup
The uniform soup recipe is incredibly simple. It consists of averaging the parameters of all
models in an ensemble.

=== Greedy Soup
The greedy soup recipe is simply a greedy algorithm: for each model in the ensemble, average its
parameters into the soup if and only if performance would be improved by doing so. Since each step
requires evaluating the performance of the potential new soup, this recipe takes much more time to
run.

= Limitations and Societal Impact

== Limitations
While the model soups approach is capable of producing improved models, it requires ensemble
training of many models in order to do so. This drastically increases the computational cost of
training when compared to finely tuning a single model. Applying this method to LDMs mitigates this
drawback to some extent, as the parameter count is reduced through the use of latent
representations, but the time and computational resources involved in training will likely remain
much greater than finely training a single probabilistic diffusion model or generative adversarial
network. Additionally, creating a model soup does nothing to mitigate the limitations inherent to
the LDM structure: sequential denoising remains much slower for inference than other model
architectures.

Another limitation is that the noisy loss graphs that diffusion models generate makes it
impossible to sort models based on a metric. Sorting of models based on performance is necessary to
begin the greedy soup algorithm. Since the loss graph is noisy, adding a model to the soup is not
gauranteed to increase performance. Thus, though greedy soups were attempted, uniform souping was
the method of choice.

== Societal Impact
Much has been discussed on the societal impact of generative machine learning models. Models
capable of generating images can be used for artistic purposes just as easily as they can be
applied to manipulate people with misinformation. While our mini-LDMs are not likely capable of
such tasks, an interested party with sufficient computational resources could apply this method
to societal benefit or detriment. The model soups approach does nothing to address these concerns.

= Results

== Uniform Soup Diffusion Models
Each individual model was evaluated on the same subset of ImageNet's validation set. Surprisingly,
each model produced the exact same loss on the validation subset. We checked the weights and biases
of each of the networks and there was enough variation between each networks weights and biases that 

== Future Work
There are many possible applications of the model soups approach that remain unexplored in this
paper. It is possible that we would have seen a greater improvement, for example, if we had reused
autoencoder parameters from an existing finely-trained LDM, thus providing a highly accurate latent
space to explore by training the remaining parameters of the model. Another possible application
would be finely training many models on small subsets of the dataset, instead of the coarse
training we do here.

== Conclusion
