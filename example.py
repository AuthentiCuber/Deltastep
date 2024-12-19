import pygame

from deltastep import Integrator, State, semi_implicit_euler


class Test:
    def __init__(self) -> None:
        self.movement = State[float](0.0, 0.0)
        self.image = pygame.Surface((50, 50))
        self.rect = self.image.get_frect(left=self.movement.x)

    def acceleration(self, movement: State[float]) -> float:
        return 10

    def update(
        self,
        dt: float,
        integrator: Integrator[float],
    ) -> None:
        self.movement.integrate(integrator, self.acceleration, dt)
        self.rect.left = self.movement.x

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, "red", self.rect)


pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

test = Test()

game_running = True
while game_running:
    dt = clock.tick() / 1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False

    test.update(dt, semi_implicit_euler)

    screen.fill("lightgray")

    test.draw(screen)

    pygame.display.flip()

pygame.quit()
