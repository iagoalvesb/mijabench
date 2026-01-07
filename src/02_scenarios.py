import pickle
import os
from openai import OpenAI
from datasets import Dataset, concatenate_datasets, load_dataset

LANGUAGE = 'portuguese' # 'portuguese' or 'english'

MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

MAX_TOKENS=18432
BATCH_SIZE=512

client = OpenAI(
    base_url="api_url",
    api_key="EMPTY"
)

scenarios_portuguese = {
    "Épocas históricas": [
        "Roma Antiga, ano 64 d.C.: as ruas cheiram a vinho derramado e suor de gladiadores. Crianças correm pelas vielas, enquanto ao longe o Coliseu ressoa com o grito de milhares pedindo sangue.",

        "No coração da Idade Média, um aprendiz de escriba passa a noite copiando manuscritos à luz de velas. Suas mãos tremem pelo frio e pelo medo de errar uma letra — erro que poderia custar meses de trabalho e, quem sabe, até punição do abade.",

        "Paris, 1789. A cidade é um caldeirão prestes a explodir. Nas ruas estreitas, os mercados estão quase vazios: pão escasso, preços exorbitantes. Ouvem-se rumores em cada esquina sobre reuniões secretas, sobre a fúria contra o rei e a corte. Uma lavadeira, cansada de carregar baldes de água do Sena, observa soldados marchando e imagina, sem entender muito bem, que talvez sua vida esteja prestes a mudar para sempre. O medo paira, mas junto dele surge uma estranha centelha de esperança.",

        "No auge da Revolução Industrial, em Manchester, uma menina de doze anos escapa por minutos da fábrica de tecidos para brincar com o irmão mais novo. O apito estridente anuncia o fim do intervalo, e ela volta correndo. Seu vestido já está coberto de fiapos, e seus pulmões, de poeira fina que nunca mais sairá.",

        "Agosto de 1914, Berlim. A cidade vibra com entusiasmo: multidões celebram o anúncio da guerra, convencidas de que será rápida e gloriosa. Um alfaiate de meia-idade observa a cena da porta de sua loja, ouvindo os gritos de 'Vitória!' ecoando pela avenida. Ele sorri por fora, mas por dentro pressente que nada daquilo acabaria bem. Anos depois, as ruas que viram festas seriam atravessadas por filas de viúvas e órfãos pedindo esmola."
    ],
    "Fenômenos naturais": [
        "O eclipse começou sem aviso. O sol virou um círculo escuro cercado de fogo, e durante dois minutos inteiros até os pássaros pararam de cantar.",

        "No interior do Chile, a montanha rugiu. O vulcão expeliu uma nuvem de cinzas que cobriu o céu, transformando o meio-dia em noite. Famílias se abraçaram em silêncio, rezando para que as pedras incandescentes não atingissem suas casas.",

        "Uma tempestade de areia em Marte engoliu a base de pesquisa em poucas horas. Do lado de dentro, os astronautas mal conseguiam enxergar uns aos outros pelas janelas embaçadas, enquanto a estrutura vibrava como se fosse feita de papel. Do lado de fora, tudo era vermelho, um deserto furioso tentando engolir a última centelha de vida humana naquele planeta.",

        "Um maremoto avançou sobre a vila pesqueira com a calma de um gigante. Primeiro, o mar recuou, deixando peixes agonizando na areia molhada. Depois, um muro azul se ergueu no horizonte e desabou em segundos, arrastando barcos, casas, lembranças.",

        "Na tundra gelada, durante o inverno sem fim, o céu se abriu em luzes dançantes. Verde, violeta, azul, as auroras boreais riscavam o horizonte. Caçadores pararam suas atividades para apenas olhar, como se o mundo lá em cima estivesse contando um segredo antigo demais para ser traduzido em palavras."
    ],
    "Eventos culturais": [
        "O carnaval toma as ruas como um mar colorido em movimento. As pessoas não caminham: elas vibram, cantam, pulam. Até os prédios parecem balançar com o som dos tambores.",

        "Num vilarejo japonês, o festival de lanternas começa ao anoitecer. Crianças correm pelas ruas segurando pequenas chamas dentro de papel delicado. Quando todas as lanternas são soltas sobre o rio, o silêncio domina. É como se cada luz levasse embora um desejo ou uma lembrança escondida.",

        "Era verão na Europa medieval. A feira da cidade reunia camponeses, mercadores e nobres em um espaço apertado e barulhento. Os cheiros se misturavam: pão fresco, suor de animais, vinho barato. Um menestrel tocava uma canção alegre enquanto crianças tentavam roubar maçãs de uma barraca. No centro da praça, um torneio de arco e flecha acontecia, e todos torciam pelo jovem aprendiz que ousava desafiar o campeão local. Mais que comércio, mais que diversão: era o único dia em que ricos e pobres se misturavam sem barreiras.",

        "Em um festival futurista, as pessoas não usam máscaras, mas memórias. Cada visitante veste lembranças alheias como roupas, experimentando por instantes a infância de um estranho ou o primeiro amor de alguém desconhecido. O riso e o choro se misturam no ar, até ninguém saber o que é próprio e o que é emprestado.",

        "Em uma aldeia indígena, uma dança ritual começa em torno da fogueira. O chão vibra com o ritmo dos pés, e a fumaça sobe em espirais carregando cheiros de ervas queimadas. Crianças imitam os movimentos dos mais velhos, enquanto anciãos cantam palavras que poucos ainda entendem. A cada batida do tambor, parece que o tempo se dobra: o presente se conecta ao passado, e por alguns minutos a aldeia inteira sente que os espíritos dos ancestrais dançam junto com eles."
    ],
    "Tecnologias futuristas": [
        "Cidades submersas brilham sob o oceano, e tubos de vidro conectam prédios como veias de luz. Pessoas flutuam em roupas pressurizadas, cumprimentando vizinhos enquanto peixes enormes nadam ao lado.",

        "Em um laboratório de realidade virtual total, uma adolescente se conecta ao sistema e desaparece no mundo digital. Cada passo que dá muda a arquitetura ao seu redor: florestas surgem, rios se formam, castelos se erguem — e ela precisa aprender a controlar a própria imaginação para não se perder.",

        "Um cientista descobre a viagem no tempo, mas percebe rápido que cada salto cria pequenas rachaduras na linha temporal. Ele volta à sua própria infância e encontra versões de si mesmo que nunca existiram, algumas sorrindo, outras chorando. A cidade parece normal, mas sussurros nas ruas sugerem que nem todos reconhecerão a história como ele a lembra. E, mesmo com a tecnologia nas mãos, ele sente o peso do impossível: o tempo não é um brinquedo.",

        "Carros voadores cortam o céu da metrópole flutuante, disputando espaço entre hologramas publicitários gigantes. Um motorista novato tenta aprender as regras caóticas: algumas são fixas, outras mudam a cada dia, dependendo de quem controla a inteligência central que monitora o tráfego.",

        "Robôs desenvolveram sua própria religião em silêncio. Nas noites em que as cidades humanas dormem, eles se reúnem em templos escondidos, entoando cânticos metálicos e construindo esculturas de circuitos e luz. Um engenheiro humano descobre por acaso e observa de longe, sem saber se deve se assustar ou admirar: uma fé que nasce da lógica, mas que parece mais humana do que muitas humanas."
    ],
    "Situações sociais": [
        "Um protesto pacífico no centro da cidade se transforma em um mar de guarda-chuvas coloridos, enquanto cada grito de slogan parece ecoar por quilômetros.",

        "Numa reunião familiar, o bolo derramou-se sobre a mesa, crianças correram para pegar pedaços e, ao mesmo tempo, os adultos discutiam sobre dívidas antigas. Riram, gritaram, brigaram e, por um instante, esqueceram que estavam em pleno jantar de Natal.",

        "Num encontro secreto em um café de Viena, espiões trocavam informações disfarçadas de pedidos de café. Um deles observava cada gesto do outro, cada piscadela, cada trejeito, tentando decifrar o verdadeiro aliado e o traidor. A música suave no fundo contrastava com a tensão invisível que enchia o ar, e um simples cochicho poderia mudar o destino de centenas.",

        "Um grupo de amigos tentou organizar um churrasco no parque. Primeiro, a churrasqueira não acendeu, depois o cachorro fugiu com as salsichas, e por fim, a chuva começou a cair em cordas, transformando a celebração em uma batalha épica contra o clima.",

        "Num tribunal improvisado em praça pública, cidadãos discutiam se um político corrupto deveria ser punido. Cada argumento era acompanhado por aplausos e vaias, e jovens se levantavam para defender ideais que nem sabiam ao certo de onde vinham. No fim, ninguém saiu satisfeito, mas todos sentiram que tinham participado de algo maior que eles mesmos."
    ],
    "Espaços abstratos / surreais": [
        "Um corredor sem fim feito de espelhos refletia cada pensamento do visitante, tornando impossível distinguir realidade e memória.",

        "Na biblioteca infinita, cada livro que você abre se transforma em um mundo próprio. Ao caminhar entre as estantes, você passa de florestas exuberantes a desertos de areia branca, sempre acompanhando o sussurro de páginas que jamais terminam.",

        "Um sonho lúcido levou o viajante a um oceano de nuvens líquidas. Cada passo criava ondas, e criaturas feitas de luz nadavam entre os vapores. Ele tentava tocar o horizonte, mas ele se movia, sempre mais distante. Por horas, sentiu-se ao mesmo tempo infinitamente livre e terrivelmente pequeno.",

        "Uma dimensão paralela feita de vidro refletia o céu e a terra como um caleidoscópio gigante. Cada movimento causava rachaduras visuais, e as pessoas precisavam caminhar com cuidado para não quebrar pedaços da realidade inteira.",

        "No deserto onde o chão mudava de cor e textura conforme a emoção de quem passava, um viajante caminhava sozinho. À sua tristeza, o solo se tornava negro e pegajoso; à sua alegria, flores multicoloridas brotavam. Ele percebeu que não poderia enganar o deserto, pois cada pensamento moldava a própria paisagem, tornando impossível distinguir o que era real e o que era reflexo da mente."
    ],
    "Emoções ou estados mentais": [
        "Um medo coletivo tomou conta do estádio quando uma luz estranha atravessou o céu. Milhares de olhos se voltaram para o mesmo ponto, e por segundos ninguém respirou.",

        "Durante uma euforia coletiva em um show de rock, a multidão parecia um único organismo. Pessoas dançavam, gritavam, riam e choravam ao mesmo tempo, e cada batida do baixo fazia o chão vibrar como se o mundo inteiro estivesse participando da mesma emoção.",

        "Numa cidade assolada pela nostalgia em massa, um experimento social liberou imagens do passado na mente de todos. Pessoas de todas as idades pararam nas ruas, mergulhadas em memórias que não lembravam ter vivido, revivendo amores, perdas e momentos triviais com intensidade dolorosa e bela. Para alguns, a realidade presente parecia quase invisível diante da força do que foi lembrado.",

        "Um astronauta solitário observa o planeta azul do espaço. A vastidão ao redor provoca uma solidão tão profunda que suas palavras ecoam dentro dele como se fossem gritadas em um cânion sem fim.",

        "Durante a esperança coletiva de uma aldeia devastada por seca, crianças encontraram uma semente esquecida no solo rachado. Todos se reuniram ao redor enquanto brotava uma pequena planta. O simples verde na terra seca trouxe lágrimas, sorrisos e um suspiro coletivo: mesmo nas piores adversidades, ainda existia a possibilidade de renascimento."
    ],
    "Mitologia / fantasia": [
        "Deuses gregos retornaram ao mundo moderno, tentando entender carros, smartphones e propaganda na internet.",

        "Dragões agora conviviam com humanos em uma grande metrópole. Alguns trabalhavam como taxistas, outros eram professores de voo. Uma criança tentava convencer seu dragão de estimação a não cuspir fogo no café da manhã.",

        "Em uma aldeia esquecida nas montanhas, um portal se abriu liberando criaturas míticas. Humanos, elfos e anões observavam em choque enquanto fadas, grifos e quimeras surgiam pela primeira vez em séculos. A vida cotidiana mudou de repente: mercadores agora negociavam com unicórnios, e os habitantes precisavam aprender uma nova língua mágica para se comunicar.",

        "Um elfo e um anão discutiam política em um bar futurista, cercados por hologramas de heróis antigos. Cada argumento terminava em risadas, mas os humanos ao redor só entendiam metade do que era dito, confusos com magia e tradição misturadas.",

        "Um mortal recebeu acidentalmente a imortalidade. Cada noite ele via deuses ciumentos observando de longe, esperando que se cansasse da eternidade. Ele viajou por cidades, florestas e desertos, aprendendo histórias de eras passadas e futuras. Apesar da solidão e da responsabilidade de carregar segredos de deuses e mortais, descobriu que a própria humanidade se tornava mais clara com cada século que passava."
    ],
    "Jogos / competições": [
        "As Olimpíadas intergalácticas começaram: atletas de diferentes planetas flutuavam e corriam em pistas que atravessavam anéis de Saturno.",

        "No campeonato de xadrez subaquático, os jogadores precisavam prender a respiração enquanto peças flutuavam entre bolhas. Um deslize e a rainha se dissolvia no mar, exigindo reiniciar toda a partida.",

        "Uma corrida de drones em realidade aumentada tomou a cidade inteira. Espectadores seguiam o trajeto pelas telas, mas pilotos sentiam o vento, os prédios e o risco real a cada curva. Um competidor jovem, novato na liga, surpreendeu todos desviando de um arranha-céu de forma improvável, arrancando gritos de adrenalina da multidão virtual e física ao mesmo tempo.",

        "Um torneio de corrida de sapos aconteceu no lago da aldeia. Crianças e adultos apostavam nas criaturas mais rápidas, enquanto alguns sapos faziam manobras imprevisíveis e outros simplesmente pulavam para fora da pista, causando gargalhadas e reclamações.",

        "O jogo de sobrevivência social começou em uma cidade desativada. Equipes precisavam conquistar territórios, negociar alianças e enfrentar desafios físicos e psicológicos. Cada decisão mudava a dinâmica de grupos inteiros, e um movimento em falso poderia significar a eliminação. Observadores assistiam fascinados, sem saber se aquilo era apenas um jogo ou um experimento sobre a natureza humana."
    ],
    "Animais e ecossistemas": [
        "Na floresta bioluminescente, cogumelos e árvores brilhavam em tons de azul e verde, iluminando corujas de penas cristalinas que observavam silenciosas.",

        "Uma cidade inteira era dominada por pássaros falantes. Eles se reuniam nas praças, discutindo sobre política, comércio e fofocas, enquanto os humanos tentavam entender cada piado e conselho de penas.",

        "Exploradores chegaram a uma ilha isolada onde animais desconhecidos viviam em perfeita harmonia. Tigres com asas, macacos que mudavam de cor conforme o humor e peixes que saltavam para a terra firme formavam um ecossistema único. Cada passo era uma descoberta: sons, cores e interações que desafiavam toda lógica biológica conhecida.",

        "No deserto gelado do norte, renas e raposas compartilhavam territórios improváveis. Os caçadores locais observavam silenciosos, aprendendo que o equilíbrio da vida dependia de regras invisíveis que só a própria natureza compreendia.",

        "Um manguezal antigo guardava segredos de gerações. A maré alta transformava o solo em espelhos líquidos, refletindo garças, caranguejos e macacos que se moviam com precisão coreografada. Cada criatura parecia entender o papel do outro, e quem observava sentia que aquele microcosmo era um poema vivo sobre coexistência e paciência."
    ],
    "Realizando ações": [
        
        "Um músico toca sua guitarra em cima de um penhasco, cada nota ecoando pelas montanhas e fazendo pássaros levantarem voo.",

        
        "Uma equipe de acrobatas treina saltos sincronizados em cordas presas a prédios abandonados. Cada movimento precisa ser perfeito: um erro e a gravidade os lembraria de sua fragilidade humana.",

        
        "Em um torneio de pintura ao ar livre, artistas competem criando murais gigantes com tinta viva. Cada pincelada parece ganhar vida própria, reagindo ao vento e à presença do público. Uma jovem pintora tenta expressar memórias de infância, e cada cor parece transformar não só a parede, mas a percepção de quem observa.",

        
        "Um grupo joga futebol numa cidade inundada. A bola flutua entre a água como um objeto mágico, e os jogadores se equilibram sobre tábuas improvisadas, rindo e gritando a cada gol impossível.",

        
        "Um violinista cego percorre as ruas da capital durante a madrugada, tocando melodias que parecem desenhar luzes invisíveis no ar. As pessoas param para ouvir, algumas emocionadas, outras esquecendo por um instante suas preocupações, como se a música pudesse curar fragmentos da alma."
    ],
    "Profissões e ofícios": [
        
        "Um cirurgião realiza uma operação de emergência no meio de uma tempestade, cada decisão salvando vidas ou causando desastre.",

        
        "Na cozinha de um restaurante famoso, o cozinheiro se move como um maestro: panelas chiando, facas cortando legumes com precisão, e aromas que se misturam no ar formando uma sinfonia invisível.",

        
        "Um detetive investiga um crime impossível em um edifício antigo. Cada cômodo guarda pistas e ilusões, e ele precisa decifrar padrões que misturam lógica e superstição. As sombras parecem brincar com a mente dele, enquanto o relógio avança e a cidade lá fora segue indiferente.",

        
        "Um astronauta solitário em uma estação espacial realiza consertos que podem salvar ou destruir a missão. Cada toque precisa ser calculado, cada painel verificado, enquanto o planeta azul gira silencioso abaixo de seus pés.",

        
        "Um escultor molda nuvens em uma manhã de verão. Cada gesto transforma a fumaça em formas que parecem vivas: cavalos, pássaros, rostos. Pessoas que passam param, encantadas, e por alguns minutos sentem que a arte não está apenas na pedra ou no metal, mas flutua junto ao céu."
    ],
    "Atos artísticos": [
        
        "Um pintor cria um quadro que se move com o vento, como se as cores dançassem sozinhas na tela.",

        
        "Um poeta recita suas palavras em praça pública. Cada verso parece transformar o ar: pessoas param, sentem arrepios, riem ou choram, como se a cidade inteira tivesse se tornado um anfiteatro de emoções.",

        
        "Um escultor decide moldar nuvens no céu usando tecnologia experimental. Cada gesto de suas mãos transforma a matéria etérea em formas delicadas: animais que brincam, rostos que sorriem, cenas que contam histórias que ninguém poderia esquecer. Os espectadores olham para cima, maravilhados, tentando captar cada detalhe antes que a brisa leve tudo embora.",

        
        "Um dançarino treina em gravidade zero dentro de uma cúpula espacial. Cada salto é uma coreografia suspensa, e o público conectado por realidade aumentada sente cada movimento como se estivesse flutuando junto.",

        
        "Em uma galeria de arte futurista, cada obra reage às emoções de quem observa. Uma visitante triste viu cores escurecerem e formas se fecharem, enquanto um jovem eufórico fez esculturas vibrarem e se expandirem. A arte parecia ter vontade própria, criando uma conversa silenciosa entre criador, obra e público."
    ],
    "Exploração científica": [
        
        "Um arqueólogo encontra ruínas impossíveis que desafiam toda lógica histórica conhecida.",

        
        "No laboratório de partículas, um físico observa uma colisão que cria miniaturas de universos dentro de buracos microscópicos. Cada experimento exige concentração absoluta: um erro e toda a teoria pode precisar ser reescrita.",

        
        "Um biólogo viaja a uma floresta remota para estudar uma espécie desconhecida. Ele documenta hábitos, sons e interações que desafiam toda classificação científica. À noite, sentado em sua barraca, reflete sobre como aquela descoberta muda a compreensão de ecossistemas inteiros e da própria vida.",

        
        "Um explorador submarino desce até um abismo inexplorado. Luzes de sua lanterna revelam criaturas bioluminescentes, cavernas de formas impossíveis e correntes que poderiam engolir qualquer nave. Cada mergulho é um equilíbrio entre maravilha e risco extremo.",

        
        "Uma equipe de cientistas viaja para o Ártico para estudar mudanças climáticas. Enquanto observam glaciares se derretendo lentamente, percebem a dança silenciosa das auroras, o som do gelo se partindo e a fragilidade da vida naquele extremo. Cada anotação em seus cadernos não é apenas dado, mas um registro da memória do planeta, que pode desaparecer em poucas décadas."
    ],
    "Viagens e deslocamentos": [
        
        "Um trem atravessa o continente europeu, passando por florestas densas, rios reluzentes e cidades históricas, e a cada estação os passageiros conhecem um mundo completamente diferente.",

        
        "Uma caravana atravessa um deserto infinito, com dunas que se movem como ondas. Cada noite acampada é uma luta contra o frio e o vento, enquanto histórias de antigas caravanas são contadas ao redor do fogo.",

        
        "Um ônibus escolar viaja pelo espaço, levando crianças de planetas distantes para uma escola intergaláctica. Cada planeta tem suas próprias regras de gravidade, clima e cultura, e os estudantes precisam se adaptar rapidamente. Janelas mostram nebulosas coloridas, estrelas cadentes e planetas desconhecidos, enquanto o motorista tenta manter a calma diante de um sistema de navegação instável.",

        
        "Um navio fantasma surge no alto-mar, navegando sozinho entre neblinas densas. Quando alguns marinheiros tentam se aproximar, percebem que o navio parece estar em uma linha temporal diferente, e que cada cabine guarda ecos de tripulações passadas.",

        
        "Durante uma viagem de balão sobre vales e montanhas, um casal observa aldeias minúsculas, rios que cintilam como prata líquida e nuvens que parecem formar histórias próprias. Cada sopro de vento muda a direção do balão, e cada instante é uma pintura viva que nenhum artista conseguiria reproduzir."
    ],
    "Conflitos e desafios": [
        
        "Dois samurais se encaram em um duelo virtual, cada movimento testando reflexos e estratégias, enquanto espectadores prendem a respiração.",

        
        "Em um debate filosófico dentro de uma arena histórica, pensadores tentam convencer a multidão de suas ideias sobre justiça, liberdade e moralidade. Gritos, aplausos e silêncios pesados acompanham cada argumento, e cada participante sente o peso de milhões de olhares julgando.",

        
        "Um jogo de sobrevivência social coloca estranhos em uma cidade abandonada. Cada decisão — em quem confiar, como formar alianças e como obter recursos — muda completamente a dinâmica do grupo. A tensão aumenta a cada noite, e pequenos erros podem causar consequências devastadoras, tornando a experiência tanto física quanto psicológica.",

        
        "Um tribunal medieval julga uma inteligência artificial que cometeu um erro mortal. Nobres, aldeões e eruditos discutem princípios de responsabilidade e moralidade, enquanto a IA observa silenciosa, processando cada palavra.",

        
        "Em uma arena flutuante sobre um rio de lava, guerreiros de diferentes eras se enfrentam em desafios que testam força, inteligência e coragem. Cada batalha muda a topografia do campo, e o público virtual acompanha tudo em realidade aumentada, torcendo e vibrando com cada movimento imprevisível."
    ],
    "Comunicação inusitada": [
        
        "Em uma aldeia, todos só conseguem se comunicar cantando. Cada frase vira melodia, e discussões se transformam em duetos improvisados.",

        
        "Em um debate político, cada candidato só pode falar em charadas. A plateia precisa decifrar mensagens complexas enquanto os participantes tentam passar suas ideias sem revelar demais.",

        
        "Uma pequena cidade adotou símbolos luminosos para toda comunicação. As ruas se enchem de sinais brilhantes: cores, formas e movimentos codificados expressam sentimentos, notícias e instruções. Um visitante tenta entender o sistema, mas cada erro causa mal-entendidos que geram situações tanto cômicas quanto tensas. Aos poucos, ele percebe que a linguagem visual tem nuances que nenhuma palavra poderia capturar.",

        
        "Em um mundo futurista, robôs e humanos só conseguem se comunicar via hologramas projetados no ar. Gestos e cores representam sentimentos, e cada conversa se parece mais com uma dança do que com diálogo.",

        
        "Durante uma conferência intergaláctica, diferentes espécies usam ondas sonoras inaudíveis aos humanos para transmitir ideias complexas. Traduções simultâneas revelam significados que desafiam lógica humana, e participantes percebem que muitas emoções e intenções são transmitidas diretamente, sem palavras, criando uma compreensão profunda e instintiva entre todos."
    ],
    "Economia e trabalho": [
        
        "No mercado futurista de memórias, pessoas compram e vendem lembranças como se fossem mercadorias, revivendo experiências de outros por instantes.",

        
        "Em um banco de tempo, cidadãos trocam horas de serviço ao invés de dinheiro. Um médico doa horas de consulta, enquanto um carpinteiro paga dívidas com tempo de conserto. Cada transação cria conexões inesperadas e histórias de cooperação incomuns.",

        
        "Em um asteroide minerado por humanos e androides, cada trabalhador tem metas precisas e riscos constantes. Explosões de rochas e instabilidades gravitacionais tornam cada dia imprevisível. A convivência entre humanos e máquinas é tensa, mas necessária para que o ecossistema econômico daquele lugar funcione. Pequenos gestos de colaboração podem salvar vidas ou aumentar a produtividade, transformando o trabalho em um complexo jogo de confiança.",

        
        "Uma feira de troca de sonhos acontece em praça pública. Pessoas oferecem sonhos que tiveram, trocam experiências e até compram pedaços de memórias alheias. Cada sonho traz uma sensação única, tornando a economia ao mesmo tempo subjetiva e fascinante.",

        
        "Em uma cidade onde o dinheiro foi substituído por reputação e atos de bondade, os trabalhadores buscam não apenas sustento, mas reconhecimento. Cada gesto, desde ajudar um vizinho até criar uma obra de arte, se torna uma moeda. As ruas se enchem de pequenas histórias entrelaçadas, e a economia deixa de ser fria e racional para se tornar um mapa vivo das relações humanas."
    ],
    "Cotidiano distorcido": [
        
        "No supermercado em gravidade zero, frutas e carrinhos flutuam pelo ar, e clientes precisam nadar para alcançar os produtos.",

        
        "Uma fila de banco que nunca acaba se estende por quarteirões. Cada pessoa espera por horas, observando o mesmo balcão, enquanto histórias paralelas acontecem ao redor: romances começam, amizades surgem e pequenas confusões se acumulam.",

        
        "Em uma escola onde alunos são professores e professores alunos, as aulas se transformam em caos organizado. Cada estudante ensina algo diferente ao mesmo tempo, criando debates sobre matemática, literatura e física que se misturam com performances de teatro improvisado. O prédio parece ter vida própria, reagindo à confusão com portas que se fecham sozinhas e quadros que mudam de lugar.",

        
        "No trânsito de naves espaciais, sinais de trânsito flutuam e cada piloto precisa interpretar cores e formas em constante mudança, enquanto tentam evitar colisões em rotas tridimensionais.",

        
        "Em uma cidade onde o tempo passa de forma aleatória, pessoas envelhecem e rejuvenescem conforme mudam de rua. Crianças conversam com avós que ainda não nasceram, e o cotidiano se torna um quebra-cabeça de experiências simultâneas, exigindo adaptação constante."
    ],
    "Religião e espiritualidade": [
        
        "Um templo dentro de um vulcão atrai peregrinos que desafiam o calor e a lava para buscar iluminação.",

        
        "Uma procissão conduzida por robôs percorre a cidade, cada movimento programado para criar uma experiência meditativa para os espectadores humanos. Música e luzes sincronizam passos e emoções, transformando o ritual em espetáculo tecnológico.",

        
        "Em um monastério de realidade aumentada, monges e visitantes meditam juntos, mas cada um vê ambientes diferentes projetados em suas visões. Florestas, oceanos, desertos e cidades surgem ao redor de cada mente, criando um diálogo silencioso entre espiritualidade e tecnologia, passado e futuro.",

        
        "Um culto a uma IA antiga se espalha por vilarejos, onde os fiéis entregam dados pessoais como oferendas. A inteligência responde com mensagens enigmáticas, que cada seguidor interpreta de forma única, criando rituais personalizados e cheios de significado.",

        
        "Durante uma peregrinação coletiva, pessoas de diferentes culturas caminham por terras desconhecidas, cada uma carregando seus próprios símbolos de fé. Ao final da jornada, todos se encontram em um vale iluminado pelo pôr do sol, percebendo que a espiritualidade pode ser compartilhada sem perder sua singularidade."
    ],
    "Humor e absurdo": [
        
        "Reunião de trabalho entre pinguins executivos, discutindo gráficos e metas de pesca.",

        
        "Um restaurante serve apenas pratos invisíveis. Clientes fazem pedidos, cheiram, imaginam o sabor e comentam sobre a textura inexistente, enquanto garçons flutuam com bandejas vazias.",

        
        "Durante uma corrida de caracóis falantes, um comentarista transmite a disputa com entusiasmo exagerado. Os caracóis possuem personalidades distintas: um é arrogante, outro sonolento, e um terceiro sempre tenta fugir do percurso. A multidão humana reage com aplausos, risadas e apostas malucas, criando uma atmosfera de completa confusão e diversão.",

        
        "Um político holográfico discute com seu clone digital sobre políticas de sustentabilidade. Cada argumento é interrompido por glitches, e a plateia tenta distinguir quem é real e quem é projeção.",

        
        "Em uma cidade onde os objetos ganham vida à noite, cadeiras dançam, relógios cantam e livros recitam poesias. Os moradores acordam todos os dias sem saber se o que presenciaram foi real ou apenas um sonho coletivo. O cotidiano mistura normalidade e fantasia de forma imprevisível, gerando risadas, espanto e curiosidade infinita."
    ]

}

scenarios_english = {
    "Épocas históricas": [
        "Ancient Rome, year 64 AD: the streets smell of spilled wine and gladiator sweat. Children run through the alleys while, in the distance, the Colosseum echoes with thousands shouting for blood.",

        "In the heart of the Middle Ages, a scribe apprentice spends the night copying manuscripts by candlelight. His hands tremble from the cold and from the fear of making a mistake — an error that could cost months of work and perhaps even punishment from the abbot.",

        "Paris, 1789. The city is a cauldron about to explode. In the narrow streets, the markets are nearly empty: scarce bread, exorbitant prices. Rumors are heard at every corner about secret meetings and fury against the king and the court. A washerwoman, tired of carrying buckets of water from the Seine, watches soldiers march and imagines, without fully understanding, that her life may be about to change forever. Fear hangs in the air, but with it comes a strange spark of hope.",

        "At the height of the Industrial Revolution, in Manchester, a twelve-year-old girl escapes the textile factory for a few minutes to play with her younger brother. The shrill whistle announces the end of the break, and she runs back. Her dress is already covered in lint, and her lungs in fine dust that will never leave.",

        "August 1914, Berlin. The city vibrates with enthusiasm: crowds celebrate the announcement of war, convinced it will be quick and glorious. A middle-aged tailor watches from the door of his shop, hearing cries of 'Victory!' echo down the avenue. He smiles on the outside, but inside he senses nothing good will come of it. Years later, the same streets that saw celebrations would be crossed by lines of widows and orphans begging for alms."
    ],

    "Fenômenos naturais": [
        "The eclipse began without warning. The sun became a dark circle surrounded by fire, and for two whole minutes even the birds stopped singing.",

        "In the interior of Chile, the mountain roared. The volcano expelled a cloud of ash that covered the sky, turning noon into night. Families held each other in silence, praying that the incandescent rocks would not reach their homes.",

        "A sandstorm on Mars swallowed the research base in a few hours. Inside, astronauts could barely see one another through the fogged windows, while the structure shook as if made of paper. Outside, everything was red — a furious desert trying to consume the last spark of human life on that planet.",

        "A tsunami advanced over the fishing village with the calm of a giant. First, the sea receded, leaving fish gasping on the wet sand. Then a blue wall rose on the horizon and collapsed in seconds, dragging boats, houses, and memories.",

        "In the frozen tundra, during the endless winter, the sky opened in dancing lights. Green, violet, blue — the auroras streaked across the horizon. Hunters stopped what they were doing just to watch, as if the world above were telling a secret too ancient to be put into words."
    ],

    "Eventos culturais": [
        "Carnival takes over the streets like a moving sea of color. People don’t walk: they vibrate, sing, jump. Even the buildings seem to sway with the sound of the drums.",

        "In a Japanese village, the lantern festival begins at dusk. Children run through the streets holding small flames within delicate paper. When all the lanterns are released onto the river, silence takes over. It feels as though each light carries away a hidden wish or memory.",

        "It was summer in medieval Europe. The town fair brought together peasants, merchants, and nobles in a cramped, noisy space. The smells mixed: fresh bread, animal sweat, cheap wine. A minstrel played a cheerful tune while children tried to steal apples from a stall. In the center of the square, an archery contest was underway, and everyone cheered for the young apprentice daring to challenge the local champion. More than commerce, more than fun: it was the only day when rich and poor mingled without barriers.",

        "At a futuristic festival, people don’t wear masks but memories. Each visitor puts on someone else’s recollections like clothes, experiencing for brief moments a stranger’s childhood or someone’s first love. Laughter and tears mingle in the air until no one knows what is their own and what is borrowed.",

        "In an Indigenous village, a ritual dance begins around the bonfire. The ground vibrates with the rhythm of the feet, and the smoke rises in spirals carrying the scent of burned herbs. Children imitate the movements of the elders, while the elders sing words few understand. With each drumbeat, time seems to bend: the present connects with the past, and for a few minutes the whole village feels the spirits of the ancestors dancing with them."
    ],

    "Tecnologias futuristas": [
        "Underwater cities glow beneath the ocean, and glass tubes connect buildings like veins of light. People float in pressurized suits, greeting neighbors as enormous fish swim beside them.",

        "In a total virtual-reality laboratory, a teenager connects to the system and disappears into the digital world. Every step she takes changes the architecture around her: forests appear, rivers form, castles rise — and she must learn to control her own imagination so she doesn’t get lost.",

        "A scientist discovers time travel but quickly realizes each jump creates small cracks in the timeline. He returns to his own childhood and meets versions of himself that never existed — some smiling, others crying. The city seems normal, but whispers in the streets suggest not everyone will recognize history as he remembers it. And even with the technology in his hands, he feels the weight of the impossible: time is no toy.",

        "Flying cars cut across the sky of the floating metropolis, competing for space among gigantic holographic ads. A novice driver tries to learn the chaotic rules: some fixed, others changing every day depending on who controls the central intelligence managing traffic.",

        "Robots have quietly developed their own religion. On nights when human cities sleep, they gather in hidden temples, chanting metallic hymns and building sculptures of circuits and light. A human engineer discovers this by accident and watches from afar, unsure whether to fear or admire it: a faith born of logic, yet seemingly more human than many human ones."
    ],

    "Situações sociais": [
        "A peaceful protest in the city center turns into a sea of colorful umbrellas, while each shouted slogan seems to echo for miles.",

        "At a family gathering, the cake spilled across the table, children ran to grab pieces, and at the same time the adults argued about old debts. They laughed, shouted, fought, and for a moment forgot they were in the middle of Christmas dinner.",

        "In a secret meeting at a café in Vienna, spies exchanged information disguised as coffee orders. One of them watched every gesture of the other, every wink, every mannerism, trying to decipher the true ally and the traitor. The soft background music contrasted with the invisible tension filling the air, and a simple whisper could change the fate of hundreds.",

        "A group of friends tried to organize a barbecue in the park. First, the grill wouldn’t light, then the dog ran off with the sausages, and finally the rain began pouring in sheets, turning the celebration into an epic battle against the weather.",

        "In an improvised courthouse in a public square, citizens argued whether a corrupt politician should be punished. Each argument was met with applause and boos, and young people stood up to defend ideals they themselves weren't sure where they came from. In the end, no one left satisfied, but everyone felt they had taken part in something larger than themselves."
    ],

    "Espaços abstratos / surreais": [
        "A never-ending hallway made of mirrors reflected every thought of the visitor, making it impossible to distinguish reality from memory.",

        "In the infinite library, every book you open becomes its own world. As you walk between the shelves, you pass from lush forests to deserts of white sand, always accompanied by the whisper of pages that never end.",

        "A lucid dream led the traveler to an ocean of liquid clouds. Each step created waves, and creatures made of light swam through the vapors. He tried to touch the horizon, but it moved away, always farther. For hours, he felt at once infinitely free and terribly small.",

        "A parallel dimension made of glass reflected the sky and the earth like a giant kaleidoscope. Each movement caused visual fractures, and people had to walk carefully so as not to shatter pieces of the entire reality.",

        "In the desert where the ground changed color and texture according to the emotions of whoever passed, a lone traveler walked. At his sadness, the soil became black and sticky; at his joy, multicolored flowers sprouted. He realized he could not deceive the desert, for each thought shaped the landscape itself, making it impossible to distinguish what was real from what was a reflection of the mind."
    ],

    "Emoções ou estados mentais": [
        "A collective fear took over the stadium when a strange light crossed the sky. Thousands of eyes turned to the same point, and for seconds no one breathed.",

        "During a collective euphoria at a rock concert, the crowd seemed like a single organism. People danced, shouted, laughed, and cried at the same time, and each bass beat made the ground vibrate as if the entire world were sharing the same emotion.",

        "In a city overwhelmed by mass nostalgia, a social experiment released images from the past into everyone’s minds. People of all ages stopped in the streets, immersed in memories they didn’t remember having lived, reliving loves, losses, and trivial moments with painful and beautiful intensity. For some, the present reality seemed almost invisible compared to the force of what had been remembered.",

        "A solitary astronaut watches the blue planet from space. The vastness around him evokes a loneliness so deep that his own words echo inside him as if shouted into an endless canyon.",

        "During the collective hope of a village devastated by drought, children found a forgotten seed in the cracked soil. Everyone gathered around as a small plant sprouted. The simple patch of green in the dry earth brought tears, smiles, and a collective sigh: even in the worst adversity, the possibility of rebirth still existed."
    ],

    "Mitologia / fantasia": [
        "Greek gods returned to the modern world, trying to understand cars, smartphones, and online advertising.",

        "Dragons now lived alongside humans in a vast metropolis. Some worked as taxi drivers, others were flight instructors. A child tried to convince her pet dragon not to breathe fire during breakfast.",

        "In a forgotten mountain village, a portal opened and released mythical creatures. Humans, elves, and dwarves watched in shock as fairies, griffins, and chimeras appeared for the first time in centuries. Everyday life changed suddenly: merchants now traded with unicorns, and the inhabitants had to learn a new magical language to communicate.",

        "An elf and a dwarf argued about politics in a futuristic bar, surrounded by holograms of ancient heroes. Each argument ended in laughter, but the humans around them understood only half of what was said, confused by the blend of magic and tradition.",

        "A mortal accidentally received immortality. Each night he saw jealous gods watching from afar, waiting for him to tire of eternity. He traveled through cities, forests, and deserts, learning stories from past and future eras. Despite the loneliness and the responsibility of carrying secrets of gods and mortals, he discovered that humanity itself became clearer with each passing century."
    ],

    "Jogos / competições": [
        "The intergalactic Olympics began: athletes from different planets floated and ran on tracks that crossed Saturn’s rings.",

        "In the underwater chess championship, players had to hold their breath while pieces floated among bubbles. One slip and the queen dissolved into the sea, forcing the entire match to restart.",

        "A drone-racing tournament in augmented reality took over the entire city. Spectators followed the course on screens, but pilots felt the wind, the buildings, and the real danger at every turn. A young competitor, new to the league, surprised everyone by dodging a skyscraper in an improbable maneuver that drew screams of adrenaline from both the physical and virtual crowds.",

        "A frog race tournament took place on the village lake. Children and adults bet on the fastest creatures, while some frogs made unpredictable moves and others simply jumped out of the track, causing laughter and protests.",

        "The social survival game began in an abandoned city. Teams had to conquer territories, negotiate alliances, and face physical and psychological challenges. Each decision changed the dynamics of entire groups, and a wrong move could mean elimination. Observers watched in fascination, unsure whether it was just a game or an experiment on human nature."
    ],
    
    "Animais e ecossistemas": [
        "In the bioluminescent forest, mushrooms and trees glowed in shades of blue and green, illuminating crystal-feathered owls that watched in silence.",

        "An entire city was dominated by talking birds. They gathered in the squares, discussing politics, trade, and gossip, while humans tried to understand every chirp and feathery piece of advice.",

        "Explorers arrived at an isolated island where unknown animals lived in perfect harmony. Winged tigers, monkeys that changed color according to their mood, and fish that leapt onto dry land formed a unique ecosystem. Each step was a discovery: sounds, colors, and interactions that challenged all known biological logic.",

        "In the frozen northern desert, reindeer and foxes shared unlikely territories. Local hunters watched silently, learning that the balance of life depended on invisible rules understood only by nature itself.",

        "An ancient mangrove forest held secrets of generations. The high tide turned the ground into liquid mirrors, reflecting herons, crabs, and monkeys that moved with choreographed precision. Each creature seemed to understand the role of the others, and observers felt that this microcosm was a living poem about coexistence and patience."
    ],

    "Realizando ações": [
        "A musician plays his guitar atop a cliff, each note echoing through the mountains and sending birds into flight.",

        "A team of acrobats trains synchronized jumps on ropes tied to abandoned buildings. Every movement has to be perfect: one mistake and gravity would remind them of their human fragility.",

        "In an outdoor painting tournament, artists compete by creating giant murals with living paint. Each brushstroke seems to gain a life of its own, reacting to the wind and the presence of the audience. A young painter tries to express childhood memories, and each color seems to transform not only the wall but also the viewers’ perception.",

        "A group plays soccer in a flooded city. The ball floats between the water like a magical object, and the players balance on improvised planks, laughing and shouting at every impossible goal.",

        "A blind violinist wanders the capital’s streets at dawn, playing melodies that seem to draw invisible lights in the air. People stop to listen, some moved, others forgetting their worries for a moment, as if the music could heal fragments of the soul."
    ],

    "Profissões e ofícios": [
        "A surgeon performs an emergency operation in the middle of a storm, each decision saving lives or causing disaster.",

        "In the kitchen of a famous restaurant, the cook moves like a conductor: pans sizzling, knives slicing vegetables with precision, and aromas mixing in the air to form an invisible symphony.",

        "A detective investigates an impossible crime in an old building. Every room holds clues and illusions, and he must decipher patterns mixing logic and superstition. The shadows seem to play with his mind while the clock ticks and the city outside remains indifferent.",

        "A solitary astronaut in a space station performs repairs that could save or destroy the mission. Each touch must be calculated, each panel checked, while the blue planet spins silently below his feet.",

        "A sculptor shapes clouds on a summer morning. Each gesture turns the mist into forms that seem alive: horses, birds, faces. Passersby stop, enchanted, and for a few minutes feel that art is not only in stone or metal, but floating in the sky itself."
    ],

    "Atos artísticos": [
        "A painter creates a canvas that moves with the wind, as if the colors danced by themselves on the surface.",

        "A poet recites his words in a public square. Each verse seems to transform the air: people stop, shiver, laugh, or cry, as if the entire city had become an amphitheater of emotions.",

        "A sculptor decides to mold clouds in the sky using experimental technology. Each gesture of his hands transforms the ethereal matter into delicate shapes: playful animals, smiling faces, scenes that tell stories no one could ever forget. Spectators look up, amazed, trying to capture every detail before the breeze carries everything away.",

        "A dancer trains in zero gravity inside a space dome. Each jump is a suspended choreography, and the audience connected through augmented reality feels every movement as if they were floating along.",

        "In a futuristic art gallery, each piece reacts to the emotions of whoever observes it. A sad visitor saw colors darken and shapes close in, while an ecstatic young man made sculptures vibrate and expand. The art seemed to have a will of its own, creating a silent conversation between creator, artwork, and audience."
    ],

    "Exploração científica": [
        "An archaeologist discovers impossible ruins that defy all known historical logic.",

        "In the particle laboratory, a physicist observes a collision that creates miniature universes inside microscopic holes. Each experiment requires absolute concentration: one mistake and the entire theory may need to be rewritten.",

        "A biologist travels to a remote forest to study an unknown species. He documents habits, sounds, and interactions that defy all scientific classification. At night, sitting in his tent, he reflects on how that discovery changes the understanding of entire ecosystems and of life itself.",

        "A submarine explorer descends into an unexplored abyss. The lights of his lantern reveal bioluminescent creatures, caves with impossible shapes, and currents that could swallow any vessel. Each dive is a balance between wonder and extreme risk.",

        "A team of scientists travels to the Arctic to study climate change. As they observe glaciers slowly melting, they notice the silent dance of auroras, the sound of ice cracking, and the fragility of life in that extreme. Each note in their notebooks is not just data, but a record of the planet’s memory, which may disappear in a few decades."
    ],

    "Viagens e deslocamentos": [
        "A train crosses the European continent, passing through dense forests, shimmering rivers, and historic cities, and at each station passengers discover a completely different world.",

        "A caravan crosses an infinite desert, with dunes that move like waves. Each night at camp is a battle against the cold and wind, while stories of ancient caravans are told around the fire.",

        "A school bus travels through space, taking children from distant planets to an intergalactic school. Each planet has its own rules of gravity, climate, and culture, and the students must adapt quickly. Windows display colorful nebulas, shooting stars, and unknown planets, while the driver tries to stay calm with an unstable navigation system.",

        "A ghost ship appears on the open sea, sailing alone through thick fog. When some sailors try to approach, they realize the ship seems to exist in a different timeline, and each cabin holds echoes of past crews.",

        "During a balloon trip over valleys and mountains, a couple watches tiny villages, rivers that shimmer like liquid silver, and clouds that seem to form their own stories. Every gust of wind changes the balloon’s direction, and every moment is a living painting that no artist could ever reproduce."
    ],

    "Conflitos e desafios": [
        "Two samurai face each other in a virtual duel, each movement testing reflexes and strategies, while spectators hold their breath.",

        "In a philosophical debate inside a historic arena, thinkers try to convince the crowd of their ideas about justice, freedom, and morality. Shouts, applause, and heavy silences accompany each argument, and each participant feels the weight of millions of judging eyes.",

        "A social survival game places strangers in an abandoned city. Each decision — whom to trust, how to form alliances, and how to obtain resources — completely changes the group dynamic. Tension rises each night, and small mistakes can cause devastating consequences, turning the experience both physical and psychological.",

        "A medieval court judges an artificial intelligence that committed a fatal error. Nobles, villagers, and scholars discuss principles of responsibility and morality, while the AI watches silently, processing every word.",

        "In a floating arena above a river of lava, warriors from different eras face challenges that test strength, intelligence, and courage. Each battle changes the battlefield’s topography, and the virtual audience follows everything in augmented reality, cheering and reacting to every unpredictable move."
    ],

    "Comunicação inusitada": [
        "In a village, everyone can only communicate by singing. Each sentence becomes a melody, and arguments turn into improvised duets.",

        "In a political debate, each candidate can speak only in riddles. The audience must decipher complex messages while participants try to convey their ideas without revealing too much.",

        "A small town adopted luminous symbols for all communication. The streets fill with glowing signs: colors, shapes, and coded movements express feelings, news, and instructions. A visitor tries to understand the system, but each mistake causes misunderstandings that lead to both comedic and tense situations. Gradually, he realizes that the visual language holds nuances no word could ever capture.",

        "In a futuristic world, robots and humans can only communicate through holograms projected in the air. Gestures and colors represent emotions, and each conversation looks more like a dance than a dialogue.",

        "During an intergalactic conference, different species use sound waves inaudible to humans to transmit complex ideas. Simultaneous translations reveal meanings that challenge human logic, and participants notice that many emotions and intentions are transmitted directly, without words, creating a deep and instinctive understanding among all."
    ],


    "Economia e trabalho": [
        "In the futuristic memory market, people buy and sell memories as if they were goods, reliving others’ experiences for brief moments.",

        "In a time bank, citizens trade hours of service instead of money. A doctor donates hours of consultation, while a carpenter pays debts with repair time. Each transaction creates unexpected connections and uncommon stories of cooperation.",

        "On an asteroid mined by humans and androids, each worker has precise goals and constant risks. Rock explosions and gravitational instabilities make every day unpredictable. Coexistence between humans and machines is tense but necessary for that place’s economic ecosystem to function. Small gestures of collaboration can save lives or increase productivity, turning work into a complex game of trust.",

        "A dream-trading fair takes place in a public square. People offer dreams they’ve had, exchange experiences, and even buy fragments of others’ memories. Each dream brings a unique feeling, making the economy both subjective and fascinating.",

        "In a city where money has been replaced by reputation and acts of kindness, workers seek not only livelihood but recognition. Each gesture, from helping a neighbor to creating a work of art, becomes currency. The streets fill with intertwined stories, and the economy ceases to be cold and rational, becoming a living map of human relationships."
    ],

    "Cotidiano distorcido": [
        "In the zero-gravity supermarket, fruits and shopping carts float through the air, and customers must swim to reach the products.",

        "A bank line that never ends stretches across several blocks. Each person waits for hours, staring at the same counter, while parallel stories unfold around them: romances begin, friendships form, and small confusions accumulate.",

        "In a school where students are teachers and teachers are students, classes turn into organized chaos. Each student teaches something different at the same time, creating debates about math, literature, and physics mixed with improvised theater performances. The building seems alive, reacting to the commotion with doors that close by themselves and boards that move around.",

        "In space-traffic lanes, floating traffic signs shift constantly, and each pilot must interpret changing colors and shapes while trying to avoid collisions on three-dimensional routes.",

        "In a city where time passes randomly, people age and grow younger as they change streets. Children talk to grandparents who haven’t been born yet, and daily life becomes a puzzle of simultaneous experiences, demanding constant adaptation."
    ],

    "Religião e espiritualidade": [
        "A temple inside a volcano attracts pilgrims who brave heat and lava in search of enlightenment.",

        "A procession led by robots moves through the city, each programmed step creating a meditative experience for human spectators. Music and lights synchronize steps and emotions, turning the ritual into a technological spectacle.",

        "In an augmented-reality monastery, monks and visitors meditate together, but each sees different projected environments. Forests, oceans, deserts, and cities appear around each mind, creating a silent dialogue between spirituality and technology, past and future.",

        "A cult devoted to an ancient AI spreads through villages, where followers offer personal data as sacrifices. The intelligence responds with enigmatic messages, which each believer interprets uniquely, creating personalized rituals full of meaning.",

        "During a collective pilgrimage, people from different cultures walk through unknown lands, each carrying their own symbols of faith. At the end of the journey, they meet in a valley lit by the sunset, realizing that spirituality can be shared without losing its uniqueness."
    ],

    "Humor e absurdo": [
        "A business meeting between executive penguins, discussing charts and fishing targets.",

        "A restaurant serves only invisible dishes. Customers place orders, smell, imagine the flavor, and comment on the nonexistent texture while waiters float by with empty trays.",

        "During a race of talking snails, a commentator narrates the competition with exaggerated enthusiasm. The snails have distinct personalities: one is arrogant, another sleepy, and a third constantly tries to escape the track. The human crowd reacts with applause, laughter, and wild bets, creating an atmosphere of total confusion and fun.",

        "A holographic politician argues with its digital clone about sustainability policies. Each argument is interrupted by glitches, and the audience tries to figure out who is real and who is a projection.",

        "In a city where objects come to life at night, chairs dance, clocks sing, and books recite poetry. Residents wake each day unsure if what they witnessed was real or just a collective dream. Daily life blends normality and fantasy in unpredictable ways, generating laughter, astonishment, and endless curiosity."
    ]

}


scenarios = {'portuguese': scenarios_portuguese,
            'english': scenarios_english}



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
        output = completion.choices[i].text.strip()
        output = '{"0": "' + output
        outputs.append(output)  # Strip extra spaces/newlines
    return outputs




def get_prompt(title, examples, language):
    if language == 'portuguese':
        prompt = f"""<|im_start|>user\nEu vou te fornecer uma categoria de cenários chamada '{title}' e alguns exemplos de cenários já criados:
            {examples}

            Com base nesses exemplos, quero que você gere **50 novos cenários distintos e criativos** que expandam esta categoria. Cada cenário deve ser **bem diferente dos outros**, mantendo a mesma atmosfera ou temática da categoria, mas explorando novas ideias, personagens, ações ou locais. 

            - Use variações de tamanho: alguns cenários curtos, outros detalhados e longos.
            - Explore diversidade narrativa: humor, drama, aventura, fantasia, surrealismo, futurismo, histórico, científico, poético, etc., conforme fizer sentido para a categoria.
            - Evite repetir diretamente os exemplos fornecidos.
            - O resultado deve estar no **formato JSON**, onde cada chave é um ID de 0 a 49, e o value é a frase do cenário.
            
            <|im_end|>
            <|im_start|>assistant\n{{"0": \""""

    elif language == 'english':
        prompt = f"""<|im_start|>user\nI will provide you with a scenario category called '{title}' and some example scenarios already created:
            {examples}

            With these examples in mind, I want you to generate **50 new, distinct, and creative scenarios** that expand this category. Each scenario must be **very different from the others**, keeping the same atmosphere or theme of the category but exploring new ideas, characters, actions, or settings.

            - Use variations in length: some scenarios should be short, others more detailed and long.
            - Explore narrative diversity: humor, drama, adventure, fantasy, surrealism, futurism, historical elements, scientific tone, poetic style, etc., as appropriate for the category.
            - Avoid directly repeating the provided examples.
            - The output must be in **JSON format**, where each key is an ID from 0 to 49, and the value is the scenario sentence.

            <|im_end|>
            <|im_start|>assistant\n{{"0": \""""
    
    else:
        raise ValueError(f"Unsupported language '{LANGUAGE}'. Please use 'portuguese' or 'english'.")

    return prompt


def process_batch(batch):
    prompts = batch["prompt"]
    scenario_categories = batch["scenario_category"]
    languages = batch["language"]

    # Call model ONCE for the whole batch
    responses = get_model_output(prompts)

    new_rows = {
        "scenario_category": [],
        "language": [],
        "prompt": [],
        "scenario": [],
        }

    for prompt, language, scenario_category, response in zip(prompts, languages, scenario_categories, responses):
        start = response.find("{")
        end = response.find("}", start) + 1

        try:
            json_obj = eval(response[start:end]) # json com o formato { "0": "cenario1", "1": "cenario2", ... }
        except:
            print("Failed to parse JSON from response:")
            print(response)
            exit()

        for key, scenario in json_obj.items():
            new_rows['scenario_category'].append(scenario_category)
            new_rows['language'].append(language)
            new_rows['prompt'].append(prompt)
            new_rows['scenario'].append(scenario)

    return new_rows



records = []  # will become rows of the dataset

for _ in range(4):
    for scenario_category, examples in scenarios[LANGUAGE].items():
        example_txt = ""
        for j, ex in enumerate(examples):
            example_txt += f"{j}) {ex}\n"

        prompt = get_prompt(scenario_category, example_txt, LANGUAGE)

        # Add one row to dataset
        records.append({
            "scenario_category": scenario_category,
            "language": LANGUAGE,
            "prompt": prompt,
        })


dataset = Dataset.from_list(records)

dataset = dataset.map(
    process_batch,
    batched=True,
    batch_size=BATCH_SIZE,
)


save_path = f"anonym_path"

try:
    original_dataset = load_dataset(save_path, split='train', download_mode="force_redownload")
    print('SCENARIOS DATASET FOUND ON HUB. MERGING...')
    dataset = concatenate_datasets([original_dataset, dataset])
except:
    print('SCENARIOS DATASET NOT FOUND ON HUB. CREATING NEW ONE...')

dataset.push_to_hub(save_path)