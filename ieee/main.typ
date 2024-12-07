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
Diffusion models generate meaningful data from random noise through a process of sequential
denoising. Such models are trained by taking initial data and "noising" it until it resembles
white noise. The model learns the reverse path: how to denoise a random input until it resembles
an item from the training dataset. Such models are very large, with millions or billions of
parameters, and require many GPU-days to train @ho2020denoising.

Latent diffusion models differ from diffusion probabilistic models in that they operate on the
latent space representation of trained inputs instead of pixel space. This means that, compared to
diffusion probabilistic models, latent diffusion models can learn a much better understanding of
the training data with an autoencoder network of the same size. These models still take a long time
to train, but produce similar results to diffusion probabilistic models with shorter training time
@ldm.

State-of-the-art models (in image synthesis as well as other tasks) often reduce training time
through transfer learning @pan2010survey. In this process, a pre-trained model is fine-tuned on a
particular task through additional training steps @model-soups. However, several other methods have
been considered for transferring learned capabilities from one model to another. Recent research
has demonstrated that merging the parameters from an ensemble of finely tuned models can produce
improved performance on target tasks @model-soups, @matena2021merging. Multiple techniques for
merging paramaters have been evaluated, including weighted averages @matena2021merging, uniform
averages, and greedy algorithms @model-soups.

In this paper, we apply a uniform average merging strategy to a finely-tuned ensemble of latent
diffusion models.

== Paper Overview



= Background

== Related Works
In recent years, generative models for image synthesis have become a popular topic of research.
Prior to the publication of @ldm, denoising diffusion probabilistic models @ho2020denoising
provided state-of-the-art image synthesis quality, albeit with much higher training times. Other
competitors in the space included generative adversarial networks (GANs) such as @brock2018large,
variational autoencoders (VAEs) as in @child2020very, and autoregressive models such as
@chen2020generative.

Since the publication of @ldm, incredible advances in the field of noise-to-data generation have
taken place. In @esser2024scaling, Esser et. al. apply rectified flow models for improved
text-to-image synthesis. Their approach results in improved sampling time by reducing the number
of steps required to generate a sample, while maintaining the perceived quality of diffusion
models. Both @ldm and @esser2024scaling are improvements on the denoising diffusion probabilistic
models introduced in @ho2020denoising.

== Latent Diffusion
Latent Diffusion, as described in @ldm, differs from prior state-of-the-art diffusion models by
performing denoising operations in latent space rather than pixel space. This significantly reduces
the computational resources required for training when compared to diffusion models which operate
on pixel space representations. At the time of publication, latent diffusion models (LDMs) offered
state-of-the-art image inpainting and provided comp\etitive capabilities for unconditional image
generation, generative upscaling, and scene synthesis from input images with labeled regions. In
this paper we test only the image generation capabilities of LDMs.

Like established diffusion architectures, the key operation of LDMs is the diffusion process. In
this process, outputs are generated from input noise through many sequential denoising steps.
However, LDMs make two key structural changes to the diffusion probabilistic model formula. First,
instead of training on pixel space, the autoencoder network is trained on the latent space. This
allows LDMs to capture relevant information while ignoring perceptually irrelevant high-frequency
components that must be trained when operating in pixel space. Second, LDMs introduce
cross-attention layers which enable generative sampling from the learned latent space.

In this paper, we will base our methods on the `cin256` model provided in @ldm. This
class-conditional model is trained on the ImageNet dataset @deng2009imagenet. 

== Model Soups
As described in @model-soups, combining parameters from an ensemble of finely-tuned models can
produce better results than any of the individual models. The authors present two approaches:
a "Uniform Soup" recipe that simply averages the parameters of all models, and a "Greedy Soup"
recipe which uses a greedy algorithm to include only those models which improve the recipe's
performance against a test dataset. The authors of @model-soups apply this to transformer-
based models. In this paper, we instead evaluate their approach with LDMs.

The justification for doing this is that each each model used in the algorithms in @model-soups was
that each model was fine-tuned from a base model such as ViT or ALIGN. Each model is fine-tuned
with varying hyperparameters such as learning rate, batch_size, etc. This produces a wide range of models
that all share the same "optimization trajectory". In theory, this places the models in a distribution
around or towards a local minimum.

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
It is also possible that the observed differences are due to updates to the dataset since the
publication of @model-soups.

= Methods
The goal of this paper is to demonstrate the effects of applying the model soups approach to an
ensemble of finely-tuned diffusion models. We begin by tuning an ensemble of models based on the
`cin256` model provided in @ldm. Each model starts with identical parameters, but is trained for 1
additional epoch through a subset of 50000 training images with a unique combination of learn rate
and batch size. This results in 9 finely-tuned models that we combine using model soups methods 
described below and detailed in @model-soups.

== The Base Model: `cin256`
The `cin256` model 

== Fine-Tuning LDM Models


== LDM Soup Recipes


=== Uniform Soup
The uniform soup recipe is incredibly simple. It consists of averaging the parameters of all
models of a collection of models.

=== Greedy Soup
The greedy soup recipe is simply a greedy algorithm: for each model in the ensemble, average its
parameters into the soup if and only if performance would be improved by doing so. Since each step
requires evaluating the performance of the potential new soup, this recipe takes much more time to
run.

==== TODO: Talk about how we couldn't do greedy soup because we can't sort by loss

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

= Conclusion

== Limitations and Societal Impact

=== Limitations
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

=== Societal Impact
Much has been discussed on the societal impact of generative machine learning models. Models
capable of generating images can be used for artistic purposes just as easily as they can be
applied to manipulate people with misinformation. While our mini-LDMs are not likely capable of
such tasks, an interested party with sufficient computational resources could apply this method
to societal benefit or detriment. The model soups approach does nothing to address these concerns.
