<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->

[![GitHub License][license-shield]][license-url]

[![Paypal][Paypal-shield]][Paypal-url][![BuyMeACoffee][BuyMeACoffee-sheild]][BuyMeACoffee-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Shumpei-Tanaka/error-diffusion-model">
    <img src="/readme.md_assets/image.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Error Diffusion Model</h3>

  <p align="center">
    a new method: "Error Diffusion" for instead of backward is made in backward.
    <br />
    <a href="https://github.com/Shumpei-Tanaka/error-diffusion-model/issues">Report Bug</a>
    ·
    <a href="https://github.com/Shumpei-Tanaka/error-diffusion-model/issues">Request Feature</a>
  </p>
  <p align="center">
    <a href="/docs/readme-en.md">English</a> •
    <a href="/readme.md">日本語 (Japanese)</a>
  </p>
</div>

## Summary

Mr.Kaneko's method: Error Diffusion (ED) on pytorch

Mr.Kaneko is a man who made Winny.

I named my implemented layer to Monoamine Layer.

This repository has Monoamine Layer. And script for MNIST training as a using example.

original reference：[web archive (japanese)](https://web.archive.org/web/20000306212433/http://village.infoweb.ne.jp:80/~fwhz9346/ed.htm)

## New train method: Error Diffusion

### Idea

Original reference has source code of c.

But I changed it for inprement on pytorch.

The next figure which from original reference shows the position of increasing the coupling strength of each synapse during increasing and decreasing output.

![ed layer increase](https://web.archive.org/web/20000306212433im_/http://village.infoweb.ne.jp/~fwhz9346/fig1.gif)
![ed layer decrease](https://web.archive.org/web/20000306212433im_/http://village.infoweb.ne.jp/~fwhz9346/fig2.gif)

I focused in that the feedback path to each coupling is uniquely determined from the error between the output and the teaching data.

One neuron outputs either excitatory or excitatory from two input values, excitatory and inhibitory.

I define excitatory input $in_p$, inhibitory input $in_n$, excitatory weight $w$, inhibitory weight $v$, sign of itself $s$. Then output $out$ is below.
$$out = s \times (in_p * w - in_n * v)$$

The following figure illustrates this.
![alt text](/readme.md_assets/image.png)

Delta of weight $d_w$,$d_v$ is defined from diff of error $d_e$, sign of itself $s_{self}$, inputs $in_p$ and $in_n$.

$$d_w = s \times d_e \times in_p$$
$$d_v = -s \times d_e \times in_n$$

$w$ will increase when need more output.
$v$ will increase when need more less output.

The $d_e$ is transmitted directly from the output layer.

In principle, the route is fixed from the beginning, so the same values can be shared and used.

### Implementation

I implemented it as shown in the following figure.

![alt text](/readme.md_assets/image-1.png)

-   MonoamineFixedDual2RepInputLayer
    -   It creates as many duplicates of the input as needed.
    -   n_outputs: $input \times n$
-   MonoamineFixedDual2MidLayer
    -   It combines two inputs each.
    -   n_outputs: $input \div  2$
    -   Repeat this layer as many times as necessary.
-   MonoamineDual2MultiOutLayer
    -   It calc each output with each inner product.
    -   n_outputs: $output$
    -   conditions: $output \mod input = 0$

I fixed route of feedback by making that copy of input in first layer.

The activation function is bypassed except for the final layer.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

The source code is licensed MIT. See [LICENSE.md][license-url].

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

-   Shumpei-Tanaka
    -   s6.tanaka.pub@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Suppurt -->

## Say Thank You

If my works feels you helpful, I would be happy to have your support for me :D

links are below.

-   [https://www.paypal.me/s6tanaka][Paypal-url]
-   [https://www.buymeacoffee.com/s6tanaka][BuyMeACoffee-url]

[![Paypal][Paypal-shield]][Paypal-url][![BuyMeACoffee][BuyMeACoffee-sheild]][BuyMeACoffee-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[release-shield]: https://img.shields.io/github/v/release/Shumpei-Tanaka/readme-template?style=flat-squere&sort=semver
[release-url]: https://github.com/Shumpei-Tanaka/error-diffusion-model/releases/latest
[license-shield]: https://img.shields.io/github/license/Shumpei-Tanaka/readme-template?flat-squere
[license-url]: /LICENSE.md
[contributors-shield]: https://img.shields.io/github/contributors/Shumpei-Tanaka/readme-template.svg?style=flat-squere
[contributors-url]: https://github.com/Shumpei-Tanaka/error-diffusion-model/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Shumpei-Tanaka/readme-template.svg?style=flat-squere
[forks-url]: https://github.com/Shumpei-Tanaka/error-diffusion-model/network/members
[stars-shield]: https://img.shields.io/github/stars/Shumpei-Tanaka/readme-template.svg?style=flat-squere
[stars-url]: https://github.com/Shumpei-Tanaka/error-diffusion-model/stargazers
[issues-shield]: https://img.shields.io/github/issues/Shumpei-Tanaka/readme-template.svg?style=flat-squere
[issues-url]: https://github.com/Shumpei-Tanaka/error-diffusion-model/issues
[Paypal-shield]: https://img.shields.io/badge/paypal.me-s6tanaka-white?style=flat-squere&logo=paypal
[Paypal-url]: https://paypal.me/s6tanaka
[BuyMeACoffee-sheild]: https://img.shields.io/badge/buy_me_a_coffee-s6tanaka-white?style=flat-squere&logo=buymeacoffee&logocolor=#FFDD00
[BuyMeACoffee-url]: https://www.buymeacoffee.com/s6tanaka
[github-flow-url]: https://docs.github.com/en/get-started/quickstart/github-flow
[semver-url]: https://semver.org/
