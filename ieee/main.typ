#import "@preview/charged-ieee:0.1.2": ieee
#import "@preview/lovelace:0.3.0": *

#show: ieee.with(
  title: [Diffusion Soup: A Recipe for Improved Diffusion Models],
  abstract: [
    Recent advances in generative machine learning models have made it easier than ever to access
    powerful image manipulation tools such as text-to-image generation, generative upscaling, and
    damaged image inpainting. These tools are powered by diffusion models which generate novel
    outputs through a process of sequentially "denoising" inputs. In this paper, we experiment with
    the possibility of finding improved latent diffusion models by applying the Model Soups
    approach: fine-tuning a large ensemble of models and combining their parameters. Our
    proof-of-concept uses class-conditional LDMs and is trained using the ImageNet dataset.
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
to train, but produce perceptually similar results to diffusion probabilistic models within less
training time @ldm.

State-of-the-art models (in image synthesis as well as other tasks) often reduce training time
through transfer learning @pan2010survey. In this process, a pre-trained model is fine-tuned on a
particular task through additional training steps @model-soups. However, several other methods have
been considered for transferring learned capabilities from one model to another. Recent research
has demonstrated that merging the parameters from an ensemble of finely tuned models can produce
improved performance on target tasks @model-soups, @matena2021merging. Multiple techniques for
merging paramaters have been evaluated, including weighted averages @matena2021merging, uniform
averages, and greedy algorithms @model-soups.

== Paper Overview
In this paper, we apply a uniform average merging strategy to a finely-tuned ensemble of latent
diffusion models and qualitatively evaluate the performance of the resulting model. @background
provides background information on recent advances in diffusion models and the transfer learning
methods applied in this paper. @methods gives detailed descriptions of the methods applied to
create and fine-tune an ensemble of models that are in theory distributed around the same local
minimum, as well as the recipes used to combine the parameters of those models. @results reviews
the experimental results using qualitative and quantitative measures. Finally, @conclusion presents
concluding remarks, ethical considerations, and suggestions for future work.

= Background <background>

== Related Works
In recent years, generative models for image synthesis have become a popular topic of research.
Prior to the publication of @ldm, denoising diffusion probabilistic models @ho2020denoising
provided state-of-the-art image synthesis quality, albeit with much higher training times. Other
competitors in the space included generative adversarial networks (GANs) @brock2018large,
variational autoencoders (VAEs) @child2020very, and autoregressive models @chen2020generative.

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
on pixel space representations. At the time @ldm was published, latent diffusion models (LDMs)
offered state-of-the-art image inpainting and provided competitive capabilities for unconditional
image generation, generative upscaling, and scene synthesis from input images with labeled regions.
In this paper we test only the image generation capabilities of LDMs.

#figure(
    image("ldm_diagram.png", width: 100%),
    caption: [
        The diagram above (from @ldm) shows the diffusion and denoising process for LDMs. Note that
        the entire diffusion process takes place in latent space, with conditioning provided by a
        variety of mechanisms.
    ]
)

Like established diffusion architectures, the key operation of LDMs is the diffusion process. In
this process, outputs are generated from input noise through many sequential denoising steps.
However, LDMs make two key structural changes to the diffusion probabilistic model formula. First,
instead of training on pixel space, the autoencoder network is trained on the latent space. This
allows LDMs to capture relevant information while ignoring perceptually irrelevant high-frequency
components that must be trained when operating in pixel space. Second, LDMs introduce
cross-attention layers into the UNet which enable unconditional generative sampling from the
learned latent space @ldm.

In this paper, we will base our methods on the `cin256` model provided in @ldm. This
class-conditional model is trained on the ImageNet dataset @deng2009imagenet. A few samples
generated by this model are displayed in @fig-cin256-samples.

#figure(
    grid(
        columns: 2,
        image("cin256/sample_-00001.png"),
        image("cin256/sample_000000.png"),
        image("cin256/sample_000001.png"),
        image("cin256/sample_000002.png"),
    ),
    caption: [4 class-conditioned samples from the pretrained `cin256` model.]
) <fig-cin256-samples>

== Model Soups
As described in @model-soups, combining parameters from an ensemble of finely-tuned models can
produce better results than any of the individual models. The authors present two approaches: a
"Uniform Soup" recipe that simply averages the parameters of all models, and a "Greedy Soup" recipe
which uses a greedy algorithm to include only those models which improve the recipe's performance
against a test dataset. See @methods for details. The authors of @model-soups apply both of these
parameter merging strategies to Vision Transformer (ViT) models and ALIGN @jia2021align.

The basic approach presented in @model-soups is as follows:
1. #[Starting with a well-trained "base" model, perform a hyperparameter sweep (over e.g. batch
size, learn rate, number of epochs, weight decay, dropout, etc.) to produce an ensemble of new
model configurations.]
2. #[For each configuration, duplicate the base model and fine-tune by training with the new model
configuration. Save the resulting parameters of each fine-tuned model.]
3. #[Combine the parameters of all fine-tuned models using either the uniform soup recipe or the
greedy soup recipe.]
4. Evaluate the performance of the resulting "soup" model.

In theory, Step 2 produces a wide range of models that all share the same "optimization
trajectory". This places the models in a distribution near a local minimum in the error
hypersurface. Combining the parameters of these tuned models in Step 3 moves the resulting model
further along the optimization trajectory in a process similar to (but notably distinct from)
stochastic gradient descent.

Using the codebase provided, we duplicated the experiment from @model-soups. The results are shown
in @fig-soup-results. Computations were performed on an AMD Radeon 6900 XT.

#figure(
    image("soups.png"),
    caption: [
        Results from replicating the experiment posed in @model-soups. Note the slight difference
        in our results compared to theirs for the greedy soup algorithm.
    ]
) <fig-soup-results>

Note that our results vary slightly from the original experiment. We suspect, but have been unable
to verify, that this is due to differences in the low-level hardware and software used to compute
the results; the authors used an Nvidia A100 with the CUDA software stack, while we used an AMD
Radeon 6900 XT with the ROCm software stack. It is also possible that the observed differences are
due to updates to the dataset since the publication of @model-soups.

= Methods <methods>
The goal of this paper is to demonstrate the effects of applying the model soups approach to an
ensemble of finely-tuned diffusion models. We begin by tuning an ensemble of models based on the
`cin256` model provided in @ldm. Each model starts with identical parameters, but is trained for 1
additional epoch through a subset of 50,000 training images with a unique combination of learn rate
and batch size. This results in 9 finely-tuned models that we combine using model soups methods 
described below and detailed in @model-soups.

== The Base Model: `cin256`
The `cin256` model is a latent diffusion model trained for class-conditional image synthesis on the
full ImageNet dataset. According to @ldm, this model achieves a Fréchet Inception Distance (FID) 
@heusel2017fid of 7.76 on ImageNet. The model is trained over 4,279,397 training steps on an Nvidia
A100 GPU. Several class-conditioned samples from this model are displayed in @fig-cin256-samples.

== Data Preparation
Due to time constraints, we elected to use a small subset of the ImageNet dataset for training.
we randomly sampled 50,000 images from the ImageNet training set to use as training data for our
experiments and 5,000 images from the ImageNet validation set to use as validation data.

== Fine-Tuning LDMs
To create an ensemble of finely-tuned models, we first drew 50,000 random images from ImageNet to
use as a training set and another 5,000 to use as a validation set. Then, starting from the
pretrained base model, we fine-tuned each model by training it on a single epoch through the data
with a unique combination of learn rate and batch size. This produces 9 models fine-tuned with the
hyperparameter pairs shown in @fig-hyperparams. Each fine-tuned model is then evaluated against a
validation dataset and its loss is recorded.

#figure(
    table(
        columns: 3,
        table.header([*Model ID*], [*Learn Rate*], [*Batch Size*]),
        [ldm1], [1e-6], [64],
        [ldm2], [2e-6], [64],
        [ldm3], [3e-6], [64],
        [ldm4], [1e-6], [32],
        [ldm5], [2e-6], [32],
        [ldm6], [3e-6], [32],
        [ldm7], [1e-6], [16],
        [ldm8], [2e-6], [16],
        [ldm9], [3e-6], [16]
    ),
    caption: [Hyperparameters used to fine-tune LDMs]
) <fig-hyperparams>

== LDM Soup Recipes

=== Uniform Soup
The uniform soup recipe is incredibly simple. It consists of averaging the parameters of all
models of a collection of models. The algorithm is displayed in @alg-uniform. Averaging is
performed sequentially because each set of parameters is large (on the order of 5 GB) and the
system on which they are stored and processed does not have sufficient memory to open and merge all
models at the same time.

#figure(
    pseudocode-list[
        - *ALGORITHM UNIFORM_SOUP*
        - *INPUT* base_model, fine_tuned_models
        - *OUTPUT* uniform_model
        + *let* N = length(fine_tuned_models)
        + *let* uniform_model = base_model
        + *for* model *in* fine_tuned_models
            + *for* parameter *in* model.parameters
                + uniform_model[parameter] += model[parameter]/N
            + *end*
        + *end*
    ],
    caption: [
        The algorithm underlying the uniform soup recipe simply averages the parameters of all
        models in the ensemble.
    ]
) <alg-uniform>

=== Greedy Soup
The greedy soup recipe is simply a greedy algorithm: for each model in the ensemble, average its
parameters into the soup if and only if performance would be improved by doing so. Since each step
requires evaluating the performance of the potential new soup, this recipe takes much more time to
run. The algorithm is displayed in @alg-greedy.

#figure(
    pseudocode-list[
        - *ALGORITHM GREEDY_SOUP*
        - *INPUT* sorted_models, val_dataset
        - *OUTPUT* greedy_model
        + *let* greedy_model = sorted_models.pop_first()
        + *let* N = 1
        + *let* best_accuracy = greedy_model.test(val_dataset)
        + *for* model *in* sorted_models
            + *let* test_model = greedy_model
            + *for* p in model.parameters
                + test_model[p] += model[p]/(N+1)
            + *end*
            + *if* test_model.test(val_dataset) > best_accuracy
                + greedy_model = test_model
                + N++
            + *end*
        + *end*
    ],
    caption: [
        The algorithm underlying the greedy soup recipe incorporates only those models which
        would improve the overall model performance.
    ]
) <alg-greedy>

= Results <results>

== Fine-Tuned LDMs
We generated 9 model configurations with the hyperparameters shown in @fig-hyperparams using
`cin256` as a base model. Each model was trained on a single epoch through the subset of 50,000
random training images and validated against 5,000 random validation images selected from ImageNet.
The loss of each model against this validation set is displayed in @fig-losses.

#figure(
    table(
        columns: 2,
        table.header([*Model ID*], [*Validation Loss*]),
        [cin256],  [1.0004303455352783],
        [ldm1],    [1.0004701614379883],
        [ldm2],    [1.0004701614379883],
        [ldm3],    [1.0004701614379883],
        [ldm4],    [1.0003424882888794],
        [ldm5],    [1.0003424882888794],
        [ldm6],    [1.0003424882888794],
        [ldm7],    [1.0000524520874023],
        [ldm8],    [1.0000524520874023],
        [ldm9],    [1.0000524520874023],
        [uniform], [1.0004701614379883]
    ),
    caption: [Loss of each model against 5,000 validation images]
) <fig-losses>

After training, we sampled the fine-tuned models to evaluate their performance. Several generated
samples are displayed in @fig-ldm1-samples. Based on the results we suspect that there was a
problem in the training or sampling process that we were unable to identify within the project
timeline.

#figure(
    grid(
        columns: 2,
        image("ldm1/sample_-00001.png"),
        image("ldm1/sample_000000.png"),
        image("ldm1/sample_000001.png"),
        image("ldm1/sample_000002.png"),
    ),
    caption: [
        4 class-conditioned samples from the fine-tuned `ldm1` model. Each image is generated
        using conditioned sampling, where a class label is provided as input to the model.
    ]
) <fig-ldm1-samples>

While we were not able to identify the issue within the timeline of this project, we have some
guesses as to what could have caused this problem. We observed similar sampling behavior with the
base `cin256` model when we loaded its parameters into a model incorrectly. It is possible that
something similar is occuring here: either the parameters of the model are being saved to a file
incorrectly, or they are being loaded from file incorrectly. Unfortunately, we have thus far been
unable to find such an error in our code; the dictionary of parameters for each fine-tuned model
appears to share the same shape and structure as the base `cin256` model when printed to console.

== Uniform LDM Soup
We applied the uniform soup recipe to our ensemble of 9 fine-tuned models, producing the resulting
`uniform` model. Using the same sampling routine that was applied to `cin256`, we sampled the new
`uniform` model. Several samples are displayed in @fig-uniform-samples.

#figure(
    grid(
        columns: 2,
        image("uniform/sample_-00001.png"),
        image("uniform/sample_000000.png"),
        image("uniform/sample_000001.png"),
        image("uniform/sample_000002.png"),
    ),
    caption: [4 class-conditioned samples from the `uniform` model.]
) <fig-uniform-samples>

Unsurprisingly, the `uniform` model exhibits the same behavior as the finely tuned models used to
create it. While the samples appear to be perceptually identical, they have different SHA256
digests. Therefore we assume that at least some of their content differs, though it may not be
easily discernible by eye.

As with the fine-tuned models, the uniform model was evaluated against the same ImageNet validation
subset. The loss is reported in @fig-losses

== Greedy LDM Soup
Unfortunately, because the greedy soup code uses categorical cross-entropy loss as its measure of
performance, we were unable to produce an improved model using the greedy soup algorithm. This is
due to the non-convergent loss behavior of latent diffusion model training: from epoch to epoch, 
the loss does not necessarily improve. Instead, the loss graph resembles random noise more than a
smooth curve. 

It is possible that using an alternative performance metric, such as FID, would allow us to apply
the greedy soup recipe to our fine-tuned LDMs. However, within the time constraints of this project
we were unable modify the algorithm to use FID as a performance measure instead of loss.

= Results <results>

== Uniform Models
Each individual model was evaluated on the same subset of ImageNet's validation set. Surprisingly,
each model produced the exact same loss on the validation subset. We checked the weights and biases

of each of the networks and there was enough variation between each networks weights and biases that
neural networks were not the same.

When we tried to sample the original base model, cin256, good samples were generated.
When we tried to sample our fine-tuned models, bad samples, all of which were the squares pictured above
were generated. This leads us to believe there was a problem in the training process. This could also
explain the bad loss results, since each image is always the same square, the loss always results in being
roughly the same value.

Why this is occuring is unclear. When we cloned the latent-diffusion repository, there were library changes
we made in order to get this to work. It's possible that training with the latest versions of these
library

= Conclusion <conclusion>
Due to issues encountered during fine-tuning, we are unable to definitively determine whether the
merging strategies proposed in @model-soups are effective for improving latent diffusion models.
Because we were unable to produce fine-tuned models that were performing as intended, we are
uncertain that these models share a similar optimization trajectory. Therefore we are uncertain
whether either of the soup recipes proposed will produce an improved model. While we hypothesize
that the model soups approach should result in improved models given good fine-tuned models as a
starting point, we are unable to experimentally determine whether this is the case. The loss data
shown in @fig-losses would seem to suggest that the model soups approach does not improve model
performance, but we do not consider this to be a valid result because the fine-tuned models do not
produce perceptually valid samples.

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
guaranteed to increase performance. Thus, though a greedy soup was attempted, uniform soup was
the recipe of choice. Switching to FID as a performance measure could alleviate this issue.

== Societal Impact
Much has been discussed on the societal impact of generative machine learning models. Models
capable of generating images can be used for artistic purposes just as easily as they can be
applied to manipulate people with misinformation. While the LDMs trained in this paper are to the
best of our knowledge incapable of such tasks, an interested party with sufficient computational
resources could apply this method to societal benefit or detriment. The model soups approach does
nothing to address these concerns, though it also does little to arm bad actors.

== Future Work
There are many possible applications of the model soups approach that remain unexplored in this
paper. It is possible that we would have seen a greater improvement, for example, if we had frozen
the autoencoder parameters from the `cin256` base model, thus providing a highly accurate latent
space to explore by training the remaining parameters of the model. Another possible avenue of
exploration is applying alternate parameter merging strategies, such as the Fisher-weighted
averaging strategy proposed in @matena2021merging.
