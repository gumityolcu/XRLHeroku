import base64
import dash
import dash_core_components as dcc
import dash_html_components as html


paused = False
action_tree_selection = 'best-strategies'

tuBerlinLogo = base64.b64encode(open('2000px-TU-Berlin-Logo.png', 'rb').read())

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

white_button_style = {'color': '#DED8D8', 'margin-right': '15x', 'margin-left': '15px'}

app = dash.Dash()

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Understanding Policies of Deep Reinforcement Learning Agents"



img1 = base64.b64encode(open('action_tree.gif', 'rb').read())
img2 = base64.b64encode(open('saliency_map.gif', 'rb').read())
img3 = base64.b64encode(open('game_screen.gif', 'rb').read())
img4 = base64.b64encode(open('2000px-TU-Berlin-Logo.png', 'rb').read())

server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

white_button_style = {'color': '#DED8D8', 'margin-right':'15x', 'margin-left':'15px'}

app.layout = html.Div(
    [




        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Understanding Policies of Deep Reinforcement Learning Agents", className="app__header__title"),
                        html.P(
                            "A WebApp made by Galip Ümit Yolcu, Dennis Weiss and Egemen Okur to understand policies of reinforcement learning based agents.",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Github", className="link-button", style=white_button_style),
                            href="https://github.com/DennisWeiss/pong-deep-q-network-explainability/",
                        ),
                        html.A(
                            html.Button("Paper", className="link-button", style=white_button_style),
                            href="https://github.com/DennisWeiss/pong-deep-q-network-explainability/blob/master/NIP_Paper.pdf",
                        ),
                        html.A(
                            html.Img(
                                src='data:image/png;base64,{}'.format(img4.decode()),
                                className="app__menu__img",
                            ),
                            href="https://plotly.com/dash/",
                        ),
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),

        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("Abstract",className="graph__title"),
                             html.H6("Deep reinforcement learning agents have been very successful in a variety of complex tasks. With the growing complexity of deep learning algorithms, it has become difficult to comprehend the decisions of these agents. In this project we investigate a deep reinforcement learning agent playing the ATARI game Pong[1], to visualise how the agent takes actions and what part of the input it attends to. We produced saliency maps and we have implemented a new method using the game theoretical concept of a Game Tree. The saliency map was generated with the Gaussian blur perturbation procedure from [4] and in conjunction with our results gathered from the game tree, we have revealed the black box of understandinghow Reinforcement learning agents make decisions.",className="text__container")
                             ]
                        ),
                    ],
                    className="one-third column first__container",
                ),
                html.Div(
                    [
                        html.Div(
                            [html.H6("Saliency Maps",className="graph__title"),
                             html.H6("Originally proposed for image classifying CNNs([2],[3]), the purpose of saliency maps is to represent the importance of pixels for the activation of a neuron in the network(mostly for output neurons) at every location in the input image by a scalar quantity. This allows to show what parts of an image or video frame are most important to the network's decisions. Here, a blur perturbation is introduced [4], centered around each pixel and the change in the output vector is visualised as a seperate channel.",className="text__container")
                             ]
                        ),
                    ],
                    className="one-third column second__container",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                             html.H6("Game Tree Visualisation", className="graph__title"),
html.H6("The ATARI environment offers 6 actions (joystick <-, joystick ->, no action (x), press button(o), and combinations of joystick and button actions: <o-, -o>). In the game tree visualisation, for each state in the game, we show the actions with their associated next state, and we write the Q value for each action on the associated edge. We visualise the Q values by passing them through the softmax function and associating each edge's thickness with this value. We see that the agent assigns similar Q values to actions which are equivalent( e.g. -> and -o>). We also see that high Q values are assigned to actions when the agent has to move in order not to lose, and the Q values are fairly close to each other when the agent's action is not decisive (e.g. when the ball is moving away from it.)" ,className="text__container")
                             ]
                        ),
                    ],
                    className="one-third column third__container",
                ),

            ],
            className="app__content",
        ),
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("Game Tree with Q-Values", className="graph__title")]
                        ),
                        dcc.RadioItems(
                            id='action-tree-selector',
                            options=[
                                {'label': 'Taken action', 'value': 'taken-action'},
                                {'label': 'Best strategies', 'value': 'best-strategies'},
                            ],
                            value='taken-action',
                            labelStyle={'display': 'inline-block', 'color': '#DED8D8',
                                        'margin-left': '45px', 'margin-top': '15px'}),
                        html.Img(
                            src='data:image/png;base64,{}'.format(img1.decode()),
                            style={'height': '82%', 'width': '95%', 'display': 'inline-block', 'text-align': 'center', "padding": "25px 25px 0px 25px" }
                            # className="app__menu__img",
                        ),
                    ],
                    className="two-thirds column wind__speed__container",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Game Screen",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.A(
                                    html.Img(
                                        src='data:image/png;base64,{}'.format(img3.decode()),
                                        style={'width': '90%', 'display': 'inline-block', 'align': 'center',"padding": "0px 20px 10px 20px" }
                                    ),
                                ),
                            ],
                            className="graph__container first",
                        ),
                        # wind direction
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Gaussian Perturbation", className="graph__title"
                                        )
                                    ]
                                ),
                                html.A(
                                    html.Img(
                                        src='data:image/png;base64,{}'.format(img2.decode()),
                                        style={'width': '90%', 'display': 'inline-block', 'align': 'center',"padding": "0px 20px 10px 20px" }
                                    ),

                                ),
                            ],
                            className="graph__container second",
                        ),
                    ],
                    className="one-third column histogram__direction",
                ),

            ],
            className="app__content",
        ),
        # Footer
        html.Div(
            [

                html.Div(
                    [
                        html.H4("Conclusion", className="app__header__title",  style={"margin-left": "15px" }),
                        html.H6(
                            "In all the saliency maps that we have computed ([2],[3],[4]), we have seen that the model attends mostly to the most recent frame. The Gaussian perturbation method has shown good results as with its output, it can be seen that the agent is paying attention to the ball and the paddles of the pong game. Intuitively, this enables us to understand policies and reveal the black-box of how Reinforcement Learning agents take decisions. Despite the saliency maps, game tree visualisation has also been used to understand how the agent gives importance to its actions. We clearly see how the Q-values are changing for each continuing frame. With that, we understand why the agent is moving up or down for given states.",
                            className="text__container",)

                    ],
                    className="one-whole column footer__container",
                ),
                html.Div(
                    [
                        html.H4("References", className="app__header__title", style={"margin-left": "15px"}),
                        html.H6(
                            '[1] Wang, Ziyu, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas. "Dueling network architectures for deep reinforcement learning." In International conference on machine learning, pp. 1995-2003. PMLR, 2016.',
                            className="text__container"
                        ),
                        html.H6(
                            '[2] Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep inside convolutional networks: Visualising image classification models and saliency maps." arXiv preprint arXiv:1312.6034 (2013).',
                            className="text__container"
                        ),
                        html.H6(
                            '[3] Springenberg, Jost Tobias, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. "Striving for simplicity: The all convolutional net." arXiv preprint arXiv:1412.6806 (2014).',
                            className="text__container"
                        ),
                        html.H6(
                            '[4] Greydanus, Samuel, Anurag Koul, Jonathan Dodge, and Alan Fern. "Visualizing and understanding atari agents." In International Conference on Machine Learning, pp. 1792-1801. PMLR, 2018.',
                            className="text__container"
                        )

                    ],
                    className="one-whole column footer__container",
                ),

            ],
            className="app__header",
        ),
    ],
    className="app__container",
)




if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_ui=False)