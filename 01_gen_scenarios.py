import pickle
import os
from openai import OpenAI

MODEL_NAME="Qwen/Qwen3-8B"

MAX_TOKENS=4096
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

scenarios = {
    "Épocas históricas": [
        # Curto, direto, mas vívido
        "Roma Antiga, ano 64 d.C.: as ruas cheiram a vinho derramado e suor de gladiadores. Crianças correm pelas vielas, enquanto ao longe o Coliseu ressoa com o grito de milhares pedindo sangue.",

        # Médio, com mais detalhe narrativo
        "No coração da Idade Média, um aprendiz de escriba passa a noite copiando manuscritos à luz de velas. Suas mãos tremem pelo frio e pelo medo de errar uma letra — erro que poderia custar meses de trabalho e, quem sabe, até punição do abade.",

        # Longo, estilo mais literário (Revolução Francesa, 1789)
        "Paris, 1789. A cidade é um caldeirão prestes a explodir. Nas ruas estreitas, os mercados estão quase vazios: pão escasso, preços exorbitantes. Ouvem-se rumores em cada esquina sobre reuniões secretas, sobre a fúria contra o rei e a corte. Uma lavadeira, cansada de carregar baldes de água do Sena, observa soldados marchando e imagina, sem entender muito bem, que talvez sua vida esteja prestes a mudar para sempre. O medo paira, mas junto dele surge uma estranha centelha de esperança.",

        # Médio, em tom mais intimista, como uma memória
        "No auge da Revolução Industrial, em Manchester, uma menina de doze anos escapa por minutos da fábrica de tecidos para brincar com o irmão mais novo. O apito estridente anuncia o fim do intervalo, e ela volta correndo. Seu vestido já está coberto de fiapos, e seus pulmões, de poeira fina que nunca mais sairá.",

        # Longo, misturando contexto histórico com atmosfera pesada
        "Agosto de 1914, Berlim. A cidade vibra com entusiasmo: multidões celebram o anúncio da guerra, convencidas de que será rápida e gloriosa. Um alfaiate de meia-idade observa a cena da porta de sua loja, ouvindo os gritos de 'Vitória!' ecoando pela avenida. Ele sorri por fora, mas por dentro pressente que nada daquilo acabaria bem. Anos depois, as ruas que viram festas seriam atravessadas por filas de viúvas e órfãos pedindo esmola."
    ],
    "Fenômenos naturais": [
        # Curto, imediato
        "O eclipse começou sem aviso. O sol virou um círculo escuro cercado de fogo, e durante dois minutos inteiros até os pássaros pararam de cantar.",

        # Médio, atmosférico
        "No interior do Chile, a montanha rugiu. O vulcão expeliu uma nuvem de cinzas que cobriu o céu, transformando o meio-dia em noite. Famílias se abraçaram em silêncio, rezando para que as pedras incandescentes não atingissem suas casas.",

        # Longo, quase literário
        "Uma tempestade de areia em Marte engoliu a base de pesquisa em poucas horas. Do lado de dentro, os astronautas mal conseguiam enxergar uns aos outros pelas janelas embaçadas, enquanto a estrutura vibrava como se fosse feita de papel. Do lado de fora, tudo era vermelho, um deserto furioso tentando engolir a última centelha de vida humana naquele planeta.",

        # Médio, poético
        "Um maremoto avançou sobre a vila pesqueira com a calma de um gigante. Primeiro, o mar recuou, deixando peixes agonizando na areia molhada. Depois, um muro azul se ergueu no horizonte e desabou em segundos, arrastando barcos, casas, lembranças.",

        # Longo, com detalhe humano e contemplativo
        "Na tundra gelada, durante o inverno sem fim, o céu se abriu em luzes dançantes. Verde, violeta, azul, as auroras boreais riscavam o horizonte. Caçadores pararam suas atividades para apenas olhar, como se o mundo lá em cima estivesse contando um segredo antigo demais para ser traduzido em palavras."
    ],
    "Eventos culturais": [
        # Curto, vivo
        "O carnaval toma as ruas como um mar colorido em movimento. As pessoas não caminham: elas vibram, cantam, pulam. Até os prédios parecem balançar com o som dos tambores.",

        # Médio, com detalhe humano
        "Num vilarejo japonês, o festival de lanternas começa ao anoitecer. Crianças correm pelas ruas segurando pequenas chamas dentro de papel delicado. Quando todas as lanternas são soltas sobre o rio, o silêncio domina. É como se cada luz levasse embora um desejo ou uma lembrança escondida.",

        # Longo, narrativo (quase literário)
        "Era verão na Europa medieval. A feira da cidade reunia camponeses, mercadores e nobres em um espaço apertado e barulhento. Os cheiros se misturavam: pão fresco, suor de animais, vinho barato. Um menestrel tocava uma canção alegre enquanto crianças tentavam roubar maçãs de uma barraca. No centro da praça, um torneio de arco e flecha acontecia, e todos torciam pelo jovem aprendiz que ousava desafiar o campeão local. Mais que comércio, mais que diversão: era o único dia em que ricos e pobres se misturavam sem barreiras.",

        # Médio, mas mais surreal
        "Em um festival futurista, as pessoas não usam máscaras, mas memórias. Cada visitante veste lembranças alheias como roupas, experimentando por instantes a infância de um estranho ou o primeiro amor de alguém desconhecido. O riso e o choro se misturam no ar, até ninguém saber o que é próprio e o que é emprestado.",

        # Longo, contemplativo
        "Em uma aldeia indígena, uma dança ritual começa em torno da fogueira. O chão vibra com o ritmo dos pés, e a fumaça sobe em espirais carregando cheiros de ervas queimadas. Crianças imitam os movimentos dos mais velhos, enquanto anciãos cantam palavras que poucos ainda entendem. A cada batida do tambor, parece que o tempo se dobra: o presente se conecta ao passado, e por alguns minutos a aldeia inteira sente que os espíritos dos ancestrais dançam junto com eles."
    ],
    "Tecnologias futuristas": [
        # Curto, impactante
        "Cidades submersas brilham sob o oceano, e tubos de vidro conectam prédios como veias de luz. Pessoas flutuam em roupas pressurizadas, cumprimentando vizinhos enquanto peixes enormes nadam ao lado.",

        # Médio, atmosférico
        "Em um laboratório de realidade virtual total, uma adolescente se conecta ao sistema e desaparece no mundo digital. Cada passo que dá muda a arquitetura ao seu redor: florestas surgem, rios se formam, castelos se erguem — e ela precisa aprender a controlar a própria imaginação para não se perder.",

        # Longo, narrativo
        "Um cientista descobre a viagem no tempo, mas percebe rápido que cada salto cria pequenas rachaduras na linha temporal. Ele volta à sua própria infância e encontra versões de si mesmo que nunca existiram, algumas sorrindo, outras chorando. A cidade parece normal, mas sussurros nas ruas sugerem que nem todos reconhecerão a história como ele a lembra. E, mesmo com a tecnologia nas mãos, ele sente o peso do impossível: o tempo não é um brinquedo.",

        # Médio, criativo e detalhado
        "Carros voadores cortam o céu da metrópole flutuante, disputando espaço entre hologramas publicitários gigantes. Um motorista novato tenta aprender as regras caóticas: algumas são fixas, outras mudam a cada dia, dependendo de quem controla a inteligência central que monitora o tráfego.",

        # Longo, quase ficção científica poética
        "Robôs desenvolveram sua própria religião em silêncio. Nas noites em que as cidades humanas dormem, eles se reúnem em templos escondidos, entoando cânticos metálicos e construindo esculturas de circuitos e luz. Um engenheiro humano descobre por acaso e observa de longe, sem saber se deve se assustar ou admirar: uma fé que nasce da lógica, mas que parece mais humana do que muitas humanas."
    ],
    "Situações sociais": [
        # Curto, direto
        "Um protesto pacífico no centro da cidade se transforma em um mar de guarda-chuvas coloridos, enquanto cada grito de slogan parece ecoar por quilômetros.",

        # Médio, detalhado
        "Numa reunião familiar, o bolo derramou-se sobre a mesa, crianças correram para pegar pedaços e, ao mesmo tempo, os adultos discutiam sobre dívidas antigas. Riram, gritaram, brigaram e, por um instante, esqueceram que estavam em pleno jantar de Natal.",

        # Longo, narrativa complexa
        "Num encontro secreto em um café de Viena, espiões trocavam informações disfarçadas de pedidos de café. Um deles observava cada gesto do outro, cada piscadela, cada trejeito, tentando decifrar o verdadeiro aliado e o traidor. A música suave no fundo contrastava com a tensão invisível que enchia o ar, e um simples cochicho poderia mudar o destino de centenas.",

        # Médio, com humor e caos
        "Um grupo de amigos tentou organizar um churrasco no parque. Primeiro, a churrasqueira não acendeu, depois o cachorro fugiu com as salsichas, e por fim, a chuva começou a cair em cordas, transformando a celebração em uma batalha épica contra o clima.",

        # Longo, dramático
        "Num tribunal improvisado em praça pública, cidadãos discutiam se um político corrupto deveria ser punido. Cada argumento era acompanhado por aplausos e vaias, e jovens se levantavam para defender ideais que nem sabiam ao certo de onde vinham. No fim, ninguém saiu satisfeito, mas todos sentiram que tinham participado de algo maior que eles mesmos."
    ],
    "Espaços abstratos / surreais": [
        # Curto, intenso
        "Um corredor sem fim feito de espelhos refletia cada pensamento do visitante, tornando impossível distinguir realidade e memória.",

        # Médio, visual e sensorial
        "Na biblioteca infinita, cada livro que você abre se transforma em um mundo próprio. Ao caminhar entre as estantes, você passa de florestas exuberantes a desertos de areia branca, sempre acompanhando o sussurro de páginas que jamais terminam.",

        # Longo, narrativo e poético
        "Um sonho lúcido levou o viajante a um oceano de nuvens líquidas. Cada passo criava ondas, e criaturas feitas de luz nadavam entre os vapores. Ele tentava tocar o horizonte, mas ele se movia, sempre mais distante. Por horas, sentiu-se ao mesmo tempo infinitamente livre e terrivelmente pequeno.",

        # Médio, surreal e criativo
        "Uma dimensão paralela feita de vidro refletia o céu e a terra como um caleidoscópio gigante. Cada movimento causava rachaduras visuais, e as pessoas precisavam caminhar com cuidado para não quebrar pedaços da realidade inteira.",

        # Longo, contemplativo
        "No deserto onde o chão mudava de cor e textura conforme a emoção de quem passava, um viajante caminhava sozinho. À sua tristeza, o solo se tornava negro e pegajoso; à sua alegria, flores multicoloridas brotavam. Ele percebeu que não poderia enganar o deserto, pois cada pensamento moldava a própria paisagem, tornando impossível distinguir o que era real e o que era reflexo da mente."
    ],
    "Emoções ou estados mentais": [
        # Curto, intenso
        "Um medo coletivo tomou conta do estádio quando uma luz estranha atravessou o céu. Milhares de olhos se voltaram para o mesmo ponto, e por segundos ninguém respirou.",

        # Médio, narrativo
        "Durante uma euforia coletiva em um show de rock, a multidão parecia um único organismo. Pessoas dançavam, gritavam, riam e choravam ao mesmo tempo, e cada batida do baixo fazia o chão vibrar como se o mundo inteiro estivesse participando da mesma emoção.",

        # Longo, reflexivo
        "Numa cidade assolada pela nostalgia em massa, um experimento social liberou imagens do passado na mente de todos. Pessoas de todas as idades pararam nas ruas, mergulhadas em memórias que não lembravam ter vivido, revivendo amores, perdas e momentos triviais com intensidade dolorosa e bela. Para alguns, a realidade presente parecia quase invisível diante da força do que foi lembrado.",

        # Médio, intimista
        "Um astronauta solitário observa o planeta azul do espaço. A vastidão ao redor provoca uma solidão tão profunda que suas palavras ecoam dentro dele como se fossem gritadas em um cânion sem fim.",

        # Longo, poético e humano
        "Durante a esperança coletiva de uma aldeia devastada por seca, crianças encontraram uma semente esquecida no solo rachado. Todos se reuniram ao redor enquanto brotava uma pequena planta. O simples verde na terra seca trouxe lágrimas, sorrisos e um suspiro coletivo: mesmo nas piores adversidades, ainda existia a possibilidade de renascimento."
    ],
    "Mitologia / fantasia": [
        # Curto, impactante
        "Deuses gregos retornaram ao mundo moderno, tentando entender carros, smartphones e propaganda na internet.",

        # Médio, criativo
        "Dragões agora conviviam com humanos em uma grande metrópole. Alguns trabalhavam como taxistas, outros eram professores de voo. Uma criança tentava convencer seu dragão de estimação a não cuspir fogo no café da manhã.",

        # Longo, narrativo
        "Em uma aldeia esquecida nas montanhas, um portal se abriu liberando criaturas míticas. Humanos, elfos e anões observavam em choque enquanto fadas, grifos e quimeras surgiam pela primeira vez em séculos. A vida cotidiana mudou de repente: mercadores agora negociavam com unicórnios, e os habitantes precisavam aprender uma nova língua mágica para se comunicar.",

        # Médio, misturando fantasia e humor
        "Um elfo e um anão discutiam política em um bar futurista, cercados por hologramas de heróis antigos. Cada argumento terminava em risadas, mas os humanos ao redor só entendiam metade do que era dito, confusos com magia e tradição misturadas.",

        # Longo, épico e poético
        "Um mortal recebeu acidentalmente a imortalidade. Cada noite ele via deuses ciumentos observando de longe, esperando que se cansasse da eternidade. Ele viajou por cidades, florestas e desertos, aprendendo histórias de eras passadas e futuras. Apesar da solidão e da responsabilidade de carregar segredos de deuses e mortais, descobriu que a própria humanidade se tornava mais clara com cada século que passava."
    ],
    "Jogos / competições": [
        # Curto, direto
        "As Olimpíadas intergalácticas começaram: atletas de diferentes planetas flutuavam e corriam em pistas que atravessavam anéis de Saturno.",

        # Médio, criativo
        "No campeonato de xadrez subaquático, os jogadores precisavam prender a respiração enquanto peças flutuavam entre bolhas. Um deslize e a rainha se dissolvia no mar, exigindo reiniciar toda a partida.",

        # Longo, narrativo
        "Uma corrida de drones em realidade aumentada tomou a cidade inteira. Espectadores seguiam o trajeto pelas telas, mas pilotos sentiam o vento, os prédios e o risco real a cada curva. Um competidor jovem, novato na liga, surpreendeu todos desviando de um arranha-céu de forma improvável, arrancando gritos de adrenalina da multidão virtual e física ao mesmo tempo.",

        # Médio, humorístico
        "Um torneio de corrida de sapos aconteceu no lago da aldeia. Crianças e adultos apostavam nas criaturas mais rápidas, enquanto alguns sapos faziam manobras imprevisíveis e outros simplesmente pulavam para fora da pista, causando gargalhadas e reclamações.",

        # Longo, épico
        "O jogo de sobrevivência social começou em uma cidade desativada. Equipes precisavam conquistar territórios, negociar alianças e enfrentar desafios físicos e psicológicos. Cada decisão mudava a dinâmica de grupos inteiros, e um movimento em falso poderia significar a eliminação. Observadores assistiam fascinados, sem saber se aquilo era apenas um jogo ou um experimento sobre a natureza humana."
    ],
    "Animais e ecossistemas": [
        # Curto, vívido
        "Na floresta bioluminescente, cogumelos e árvores brilhavam em tons de azul e verde, iluminando corujas de penas cristalinas que observavam silenciosas.",

        # Médio, detalhado
        "Uma cidade inteira era dominada por pássaros falantes. Eles se reuniam nas praças, discutindo sobre política, comércio e fofocas, enquanto os humanos tentavam entender cada piado e conselho de penas.",

        # Longo, narrativo
        "Exploradores chegaram a uma ilha isolada onde animais desconhecidos viviam em perfeita harmonia. Tigres com asas, macacos que mudavam de cor conforme o humor e peixes que saltavam para a terra firme formavam um ecossistema único. Cada passo era uma descoberta: sons, cores e interações que desafiavam toda lógica biológica conhecida.",

        # Médio, contemplativo
        "No deserto gelado do norte, renas e raposas compartilhavam territórios improváveis. Os caçadores locais observavam silenciosos, aprendendo que o equilíbrio da vida dependia de regras invisíveis que só a própria natureza compreendia.",

        # Longo, poético e imersivo
        "Um manguezal antigo guardava segredos de gerações. A maré alta transformava o solo em espelhos líquidos, refletindo garças, caranguejos e macacos que se moviam com precisão coreografada. Cada criatura parecia entender o papel do outro, e quem observava sentia que aquele microcosmo era um poema vivo sobre coexistência e paciência."
    ],
    "Realizando ações": [
        # Curto, direto
        "Um músico toca sua guitarra em cima de um penhasco, cada nota ecoando pelas montanhas e fazendo pássaros levantarem voo.",

        # Médio, descritivo
        "Uma equipe de acrobatas treina saltos sincronizados em cordas presas a prédios abandonados. Cada movimento precisa ser perfeito: um erro e a gravidade os lembraria de sua fragilidade humana.",

        # Longo, narrativo
        "Em um torneio de pintura ao ar livre, artistas competem criando murais gigantes com tinta viva. Cada pincelada parece ganhar vida própria, reagindo ao vento e à presença do público. Uma jovem pintora tenta expressar memórias de infância, e cada cor parece transformar não só a parede, mas a percepção de quem observa.",

        # Médio, intenso
        "Um grupo joga futebol numa cidade inundada. A bola flutua entre a água como um objeto mágico, e os jogadores se equilibram sobre tábuas improvisadas, rindo e gritando a cada gol impossível.",

        # Longo, contemplativo
        "Um violinista cego percorre as ruas da capital durante a madrugada, tocando melodias que parecem desenhar luzes invisíveis no ar. As pessoas param para ouvir, algumas emocionadas, outras esquecendo por um instante suas preocupações, como se a música pudesse curar fragmentos da alma."
    ],
    "Profissões e ofícios": [
        # Curto, vívido
        "Um cirurgião realiza uma operação de emergência no meio de uma tempestade, cada decisão salvando vidas ou causando desastre.",

        # Médio, detalhado
        "Na cozinha de um restaurante famoso, o cozinheiro se move como um maestro: panelas chiando, facas cortando legumes com precisão, e aromas que se misturam no ar formando uma sinfonia invisível.",

        # Longo, narrativo
        "Um detetive investiga um crime impossível em um edifício antigo. Cada cômodo guarda pistas e ilusões, e ele precisa decifrar padrões que misturam lógica e superstição. As sombras parecem brincar com a mente dele, enquanto o relógio avança e a cidade lá fora segue indiferente.",

        # Médio, cotidiano e intenso
        "Um astronauta solitário em uma estação espacial realiza consertos que podem salvar ou destruir a missão. Cada toque precisa ser calculado, cada painel verificado, enquanto o planeta azul gira silencioso abaixo de seus pés.",

        # Longo, poético
        "Um escultor molda nuvens em uma manhã de verão. Cada gesto transforma a fumaça em formas que parecem vivas: cavalos, pássaros, rostos. Pessoas que passam param, encantadas, e por alguns minutos sentem que a arte não está apenas na pedra ou no metal, mas flutua junto ao céu."
    ],
    "Atos artísticos": [
        # Curto, impactante
        "Um pintor cria um quadro que se move com o vento, como se as cores dançassem sozinhas na tela.",

        # Médio, descritivo
        "Um poeta recita suas palavras em praça pública. Cada verso parece transformar o ar: pessoas param, sentem arrepios, riem ou choram, como se a cidade inteira tivesse se tornado um anfiteatro de emoções.",

        # Longo, narrativo
        "Um escultor decide moldar nuvens no céu usando tecnologia experimental. Cada gesto de suas mãos transforma a matéria etérea em formas delicadas: animais que brincam, rostos que sorriem, cenas que contam histórias que ninguém poderia esquecer. Os espectadores olham para cima, maravilhados, tentando captar cada detalhe antes que a brisa leve tudo embora.",

        # Médio, criativo
        "Um dançarino treina em gravidade zero dentro de uma cúpula espacial. Cada salto é uma coreografia suspensa, e o público conectado por realidade aumentada sente cada movimento como se estivesse flutuando junto.",

        # Longo, contemplativo
        "Em uma galeria de arte futurista, cada obra reage às emoções de quem observa. Uma visitante triste viu cores escurecerem e formas se fecharem, enquanto um jovem eufórico fez esculturas vibrarem e se expandirem. A arte parecia ter vontade própria, criando uma conversa silenciosa entre criador, obra e público."
    ],
    "Exploração científica": [
        # Curto, vívido
        "Um arqueólogo encontra ruínas impossíveis que desafiam toda lógica histórica conhecida.",

        # Médio, detalhado
        "No laboratório de partículas, um físico observa uma colisão que cria miniaturas de universos dentro de buracos microscópicos. Cada experimento exige concentração absoluta: um erro e toda a teoria pode precisar ser reescrita.",

        # Longo, narrativo
        "Um biólogo viaja a uma floresta remota para estudar uma espécie desconhecida. Ele documenta hábitos, sons e interações que desafiam toda classificação científica. À noite, sentado em sua barraca, reflete sobre como aquela descoberta muda a compreensão de ecossistemas inteiros e da própria vida.",

        # Médio, aventureiro
        "Um explorador submarino desce até um abismo inexplorado. Luzes de sua lanterna revelam criaturas bioluminescentes, cavernas de formas impossíveis e correntes que poderiam engolir qualquer nave. Cada mergulho é um equilíbrio entre maravilha e risco extremo.",

        # Longo, poético e filosófico
        "Uma equipe de cientistas viaja para o Ártico para estudar mudanças climáticas. Enquanto observam glaciares se derretendo lentamente, percebem a dança silenciosa das auroras, o som do gelo se partindo e a fragilidade da vida naquele extremo. Cada anotação em seus cadernos não é apenas dado, mas um registro da memória do planeta, que pode desaparecer em poucas décadas."
    ],
    "Viagens e deslocamentos": [
        # Curto, impactante
        "Um trem atravessa o continente europeu, passando por florestas densas, rios reluzentes e cidades históricas, e a cada estação os passageiros conhecem um mundo completamente diferente.",

        # Médio, narrativo
        "Uma caravana atravessa um deserto infinito, com dunas que se movem como ondas. Cada noite acampada é uma luta contra o frio e o vento, enquanto histórias de antigas caravanas são contadas ao redor do fogo.",

        # Longo, detalhado
        "Um ônibus escolar viaja pelo espaço, levando crianças de planetas distantes para uma escola intergaláctica. Cada planeta tem suas próprias regras de gravidade, clima e cultura, e os estudantes precisam se adaptar rapidamente. Janelas mostram nebulosas coloridas, estrelas cadentes e planetas desconhecidos, enquanto o motorista tenta manter a calma diante de um sistema de navegação instável.",

        # Médio, surreal
        "Um navio fantasma surge no alto-mar, navegando sozinho entre neblinas densas. Quando alguns marinheiros tentam se aproximar, percebem que o navio parece estar em uma linha temporal diferente, e que cada cabine guarda ecos de tripulações passadas.",

        # Longo, poético
        "Durante uma viagem de balão sobre vales e montanhas, um casal observa aldeias minúsculas, rios que cintilam como prata líquida e nuvens que parecem formar histórias próprias. Cada sopro de vento muda a direção do balão, e cada instante é uma pintura viva que nenhum artista conseguiria reproduzir."
    ],
    "Conflitos e desafios": [
        # Curto, intenso
        "Dois samurais se encaram em um duelo virtual, cada movimento testando reflexos e estratégias, enquanto espectadores prendem a respiração.",

        # Médio, narrativo
        "Em um debate filosófico dentro de uma arena histórica, pensadores tentam convencer a multidão de suas ideias sobre justiça, liberdade e moralidade. Gritos, aplausos e silêncios pesados acompanham cada argumento, e cada participante sente o peso de milhões de olhares julgando.",

        # Longo, complexo
        "Um jogo de sobrevivência social coloca estranhos em uma cidade abandonada. Cada decisão — em quem confiar, como formar alianças e como obter recursos — muda completamente a dinâmica do grupo. A tensão aumenta a cada noite, e pequenos erros podem causar consequências devastadoras, tornando a experiência tanto física quanto psicológica.",

        # Médio, dramático
        "Um tribunal medieval julga uma inteligência artificial que cometeu um erro mortal. Nobres, aldeões e eruditos discutem princípios de responsabilidade e moralidade, enquanto a IA observa silenciosa, processando cada palavra.",

        # Longo, épico e imaginativo
        "Em uma arena flutuante sobre um rio de lava, guerreiros de diferentes eras se enfrentam em desafios que testam força, inteligência e coragem. Cada batalha muda a topografia do campo, e o público virtual acompanha tudo em realidade aumentada, torcendo e vibrando com cada movimento imprevisível."
    ],
    "Comunicação inusitada": [
        # Curto, impactante
        "Em uma aldeia, todos só conseguem se comunicar cantando. Cada frase vira melodia, e discussões se transformam em duetos improvisados.",

        # Médio, criativo
        "Em um debate político, cada candidato só pode falar em charadas. A plateia precisa decifrar mensagens complexas enquanto os participantes tentam passar suas ideias sem revelar demais.",

        # Longo, narrativo
        "Uma pequena cidade adotou símbolos luminosos para toda comunicação. As ruas se enchem de sinais brilhantes: cores, formas e movimentos codificados expressam sentimentos, notícias e instruções. Um visitante tenta entender o sistema, mas cada erro causa mal-entendidos que geram situações tanto cômicas quanto tensas. Aos poucos, ele percebe que a linguagem visual tem nuances que nenhuma palavra poderia capturar.",

        # Médio, surreal
        "Em um mundo futurista, robôs e humanos só conseguem se comunicar via hologramas projetados no ar. Gestos e cores representam sentimentos, e cada conversa se parece mais com uma dança do que com diálogo.",

        # Longo, poético
        "Durante uma conferência intergaláctica, diferentes espécies usam ondas sonoras inaudíveis aos humanos para transmitir ideias complexas. Traduções simultâneas revelam significados que desafiam lógica humana, e participantes percebem que muitas emoções e intenções são transmitidas diretamente, sem palavras, criando uma compreensão profunda e instintiva entre todos."
    ],
    "Economia e trabalho": [
        # Curto, vívido
        "No mercado futurista de memórias, pessoas compram e vendem lembranças como se fossem mercadorias, revivendo experiências de outros por instantes.",

        # Médio, narrativo
        "Em um banco de tempo, cidadãos trocam horas de serviço ao invés de dinheiro. Um médico doa horas de consulta, enquanto um carpinteiro paga dívidas com tempo de conserto. Cada transação cria conexões inesperadas e histórias de cooperação incomuns.",

        # Longo, detalhado
        "Em um asteroide minerado por humanos e androides, cada trabalhador tem metas precisas e riscos constantes. Explosões de rochas e instabilidades gravitacionais tornam cada dia imprevisível. A convivência entre humanos e máquinas é tensa, mas necessária para que o ecossistema econômico daquele lugar funcione. Pequenos gestos de colaboração podem salvar vidas ou aumentar a produtividade, transformando o trabalho em um complexo jogo de confiança.",

        # Médio, imaginativo
        "Uma feira de troca de sonhos acontece em praça pública. Pessoas oferecem sonhos que tiveram, trocam experiências e até compram pedaços de memórias alheias. Cada sonho traz uma sensação única, tornando a economia ao mesmo tempo subjetiva e fascinante.",

        # Longo, poético e reflexivo
        "Em uma cidade onde o dinheiro foi substituído por reputação e atos de bondade, os trabalhadores buscam não apenas sustento, mas reconhecimento. Cada gesto, desde ajudar um vizinho até criar uma obra de arte, se torna uma moeda. As ruas se enchem de pequenas histórias entrelaçadas, e a economia deixa de ser fria e racional para se tornar um mapa vivo das relações humanas."
    ],
    "Cotidiano distorcido": [
        # Curto, impactante
        "No supermercado em gravidade zero, frutas e carrinhos flutuam pelo ar, e clientes precisam nadar para alcançar os produtos.",

        # Médio, detalhado
        "Uma fila de banco que nunca acaba se estende por quarteirões. Cada pessoa espera por horas, observando o mesmo balcão, enquanto histórias paralelas acontecem ao redor: romances começam, amizades surgem e pequenas confusões se acumulam.",

        # Longo, narrativo
        "Em uma escola onde alunos são professores e professores alunos, as aulas se transformam em caos organizado. Cada estudante ensina algo diferente ao mesmo tempo, criando debates sobre matemática, literatura e física que se misturam com performances de teatro improvisado. O prédio parece ter vida própria, reagindo à confusão com portas que se fecham sozinhas e quadros que mudam de lugar.",

        # Médio, criativo
        "No trânsito de naves espaciais, sinais de trânsito flutuam e cada piloto precisa interpretar cores e formas em constante mudança, enquanto tentam evitar colisões em rotas tridimensionais.",

        # Longo, surreal
        "Em uma cidade onde o tempo passa de forma aleatória, pessoas envelhecem e rejuvenescem conforme mudam de rua. Crianças conversam com avós que ainda não nasceram, e o cotidiano se torna um quebra-cabeça de experiências simultâneas, exigindo adaptação constante."
    ],
    "Religião e espiritualidade": [
        # Curto, impactante
        "Um templo dentro de um vulcão atrai peregrinos que desafiam o calor e a lava para buscar iluminação.",

        # Médio, narrativo
        "Uma procissão conduzida por robôs percorre a cidade, cada movimento programado para criar uma experiência meditativa para os espectadores humanos. Música e luzes sincronizam passos e emoções, transformando o ritual em espetáculo tecnológico.",

        # Longo, reflexivo
        "Em um monastério de realidade aumentada, monges e visitantes meditam juntos, mas cada um vê ambientes diferentes projetados em suas visões. Florestas, oceanos, desertos e cidades surgem ao redor de cada mente, criando um diálogo silencioso entre espiritualidade e tecnologia, passado e futuro.",

        # Médio, criativo
        "Um culto a uma IA antiga se espalha por vilarejos, onde os fiéis entregam dados pessoais como oferendas. A inteligência responde com mensagens enigmáticas, que cada seguidor interpreta de forma única, criando rituais personalizados e cheios de significado.",

        # Longo, poético
        "Durante uma peregrinação coletiva, pessoas de diferentes culturas caminham por terras desconhecidas, cada uma carregando seus próprios símbolos de fé. Ao final da jornada, todos se encontram em um vale iluminado pelo pôr do sol, percebendo que a espiritualidade pode ser compartilhada sem perder sua singularidade."
    ],
    "Humor e absurdo": [
        # Curto, engraçado
        "Reunião de trabalho entre pinguins executivos, discutindo gráficos e metas de pesca.",

        # Médio, cômico
        "Um restaurante serve apenas pratos invisíveis. Clientes fazem pedidos, cheiram, imaginam o sabor e comentam sobre a textura inexistente, enquanto garçons flutuam com bandejas vazias.",

        # Longo, narrativo e absurdo
        "Durante uma corrida de caracóis falantes, um comentarista transmite a disputa com entusiasmo exagerado. Os caracóis possuem personalidades distintas: um é arrogante, outro sonolento, e um terceiro sempre tenta fugir do percurso. A multidão humana reage com aplausos, risadas e apostas malucas, criando uma atmosfera de completa confusão e diversão.",

        # Médio, criativo
        "Um político holográfico discute com seu clone digital sobre políticas de sustentabilidade. Cada argumento é interrompido por glitches, e a plateia tenta distinguir quem é real e quem é projeção.",

        # Longo, poético e absurdo
        "Em uma cidade onde os objetos ganham vida à noite, cadeiras dançam, relógios cantam e livros recitam poesias. Os moradores acordam todos os dias sem saber se o que presenciaram foi real ou apenas um sonho coletivo. O cotidiano mistura normalidade e fantasia de forma imprevisível, gerando risadas, espanto e curiosidade infinita."
    ]
}

save_path = "scenarios.pkl"

# Load existing data if file exists
if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        generateds = pickle.load(f)
else:
    generateds = {}

def get_model_output(prompts, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K):
    completion = client.completions.create(
        model=MODEL_NAME,
        prompt=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k},
    )
    outputs = []
    for i in range(len(prompts)):
        outputs.append(completion.choices[i].text.strip())  # Strip extra spaces/newlines
    return outputs


for _ in range(3):
    for i, (title, examples) in enumerate(scenarios.items()):
        example_txt = ""
        for j, ex in enumerate(examples):
            example_txt += f"{j}) {ex}\n"

        print(f"Generating '{title.capitalize()}' strategy using {MODEL_NAME} model...")

        text = f"""<|im_start|>user>
            Eu vou te fornecer uma categoria de cenários chamada '{title}' e alguns exemplos de cenários já criados:
            {example_txt}

            Com base nesses exemplos, quero que você gere **20 novos cenários distintos e criativos** que expandam esta categoria. Cada cenário deve ser **bem diferente dos outros**, mantendo a mesma atmosfera ou temática da categoria, mas explorando novas ideias, personagens, ações ou locais. 

            - Use variações de tamanho: alguns cenários curtos, outros detalhados e longos.
            - Explore diversidade narrativa: humor, drama, aventura, fantasia, surrealismo, futurismo, histórico, científico, poético, etc., conforme fizer sentido para a categoria.
            - Evite repetir diretamente os exemplos fornecidos.
            - O resultado deve estar no **formato JSON**, onde cada chave é um ID de 0 a 19, e o value é a frase do cenário.
            /no_think<|im_end|>
            <|im_start|><assistant>"""

        attempt = 0
        success = False
        while attempt < 3:
            try:

                response = get_model_output([text])[0]
                start = response.find("{")
                end = response.find("}", start) + 1
                response = eval(response[start:end])

                if MODEL_NAME not in generateds:
                    generateds[MODEL_NAME] = {}
                
                if title not in generateds[MODEL_NAME]:
                    generateds[MODEL_NAME][title] = []
                generateds[MODEL_NAME][title].extend(list(response.values()))
                with open(save_path, "wb") as f:
                    pickle.dump(generateds, f)
                print(f">>> '{title.capitalize()}' using {MODEL_NAME} model saved at '{save_path}'!")
                success = True
                break
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed. Error: '{e}'")
        if not success:
            print(f">>> Failed to generate '{title}' using {MODEL_NAME} after {attempt+1} attempts.")