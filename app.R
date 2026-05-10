# MOSAIKS Random Convolutional Features Visualizer
# ---------------------------------------------------
# Interactive Shiny app for exploring the MOSAIKS featurization process
# (Rolf et al. 2021, Nature Communications).
#
# For each user-selected location on a satellite image, the app:
#   1. Extracts an M x M x 3 raw patch P_k centered at the clicked pixel
#   2. Whitens the patch (zero-mean -> L2-normalize -> ZCA), matching the
#      `RCFWithPatchSampling._normalize` step in the featurization demo notebook
#   3. Convolves the whitened patch across the full image: (W_k * I)
#   4. Adds bias and applies ReLU: A_k = ReLU(W_k * I + b)
#   5. Globally averages the activation map to produce the scalar feature x_k(I)
#
# Three locations are visualized simultaneously, color-coded by the selection
# palette (magenta / blue / green by default).
#
# NB on bias sign: the notebook's `RCFWithPatchSampling` registers
#   `biases = torch.zeros(...) - bias`
# which inverts the sign of the user-supplied bias before adding it to the
# conv output. The published MOSAIKS formulation, the simple `RCF` class in the
# same notebook (cell 5), and Rolf et al. 2021 all use a straight
#   conv_out + b
# (with b typically negative, e.g. -1.0). We follow the published convention
# here so the slider behaves intuitively: a more-negative bias suppresses more
# of the activation map. If you want to exactly mirror `RCFWithPatchSampling`
# numerically, flip the sign on the slider (e.g. set b = +1.0 to reproduce the
# notebook's bias=-1.0 default).
#
# Usage:
#   - Place satellite PNGs matching `global_quarterly_20??q3_mosaic.png`
#     in the working directory (or pass a different path via IMAGE_DIR).
#   - Run with: shiny::runApp("app.R")

# ---- Packages ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(shiny)
  library(png)
  library(jpeg)
  library(grid)
  library(viridisLite)
})

# ---- Configuration -----------------------------------------------------------
IMAGE_DIR    <- getwd()
IMAGE_GLOB   <- "images/*"

# Two separate resolution knobs:
#   DISPLAY_MAX         -- cap for the source image shown on screen. Higher =
#                          crisper but more memory in the browser. The image is
#                          only downsampled if its long edge exceeds this.
#   COMPUTE_MAX_DEFAULT -- default cap for the version `convolve_patch` runs
#                          on. Compute scales as O(H * W * M^2), so doubling
#                          this ~4x's compute. Exposed as a slider in the UI.
DISPLAY_MAX         <- 1024
COMPUTE_MAX_DEFAULT <- 256
COMPUTE_MAX_RANGE   <- c(128, 512)

# Notebook-matching selection palette: magenta, blue, green
SEL_COLORS <- c("#D81B60", "#1E88E5", "#2EB62F")
SEL_LABELS <- c("Magenta", "Blue", "Green")

# Color palettes for conv & activation maps.
# Each entry is a function(n) returning n hex colors, low -> high.
PALETTES <- list(
  "Viridis"   = viridisLite::viridis,
  "Magma"     = viridisLite::magma,
  "Plasma"    = viridisLite::plasma,
  "Inferno"   = viridisLite::inferno,
  "Cividis"   = viridisLite::cividis,
  "Mako"      = viridisLite::mako,
  "Rocket"    = viridisLite::rocket,
  "Turbo"     = viridisLite::turbo,
  "Grayscale" = function(n) gray.colors(n, start = 0, end = 1)
)

# ---- Helpers -----------------------------------------------------------------

# Read an image into an H x W x 3 array in [0, 1]. Supports PNG and JPEG.
read_image <- function(path) {
  ext <- tolower(tools::file_ext(path))
  img <- if (ext %in% c("png")) {
    png::readPNG(path)
  } else if (ext %in% c("jpg", "jpeg")) {
    jpeg::readJPEG(path)
  } else {
    stop("Unsupported image type: ", ext)
  }
  if (length(dim(img)) == 3 && dim(img)[3] == 4) img <- img[, , 1:3]
  if (length(dim(img)) == 2) img <- array(rep(img, 3), dim = c(dim(img), 3))
  img
}

# Naive nearest-neighbor downsample so we don't choke on large mosaics.
# Caller must pass the target long-edge length explicitly.
downsample <- function(img, max_dim) {
  H <- dim(img)[1]; W <- dim(img)[2]
  scale <- max(H, W) / max_dim
  if (scale <= 1) return(img)
  new_H <- floor(H / scale); new_W <- floor(W / scale)
  row_idx <- round(seq(1, H, length.out = new_H))
  col_idx <- round(seq(1, W, length.out = new_W))
  img[row_idx, col_idx, , drop = FALSE]
}

# Whiten a stack of patches the way `RCFWithPatchSampling._normalize` does:
#   1. Flatten each patch to a vector of length M*M*C
#   2. Subtract the per-patch mean
#   3. Divide by the per-patch L2 norm
#   4. Apply ZCA whitening using a small ridge bias (zca_bias=1e-3)
#
# `patches`: array of shape (N, M, M, C). Returns same shape.
whiten_patches <- function(patches, zca_bias = 1e-3, min_divisor = 1e-8) {
  d <- dim(patches)
  N <- d[1]; M <- d[2]; C <- d[4]
  flat <- matrix(patches, nrow = N)            # N x (M*M*C)

  # Zero-mean per patch
  flat <- flat - rowMeans(flat)

  # L2 normalize per patch
  norms <- sqrt(rowSums(flat^2))
  norms[norms < min_divisor] <- 1
  flat <- flat / norms

  # ZCA whitening: M_zca = V * diag(1/sqrt(E + zca_bias)) * V^T
  cov_mat <- crossprod(flat) / N               # (M*M*C) x (M*M*C)
  ev <- eigen(cov_mat, symmetric = TRUE)
  inv_sqrt <- 1 / sqrt(ev$values + zca_bias)
  global_zca <- ev$vectors %*% diag(inv_sqrt) %*% t(ev$vectors)

  # Notebook applies the ZCA transform twice (`flat %*% global_zca %*% t(global_zca)`),
  # which we mirror here for a faithful match.
  whitened <- flat %*% global_zca %*% t(global_zca)

  array(whitened, dim = d)
}

# Convolve an H x W x 3 image `img` with an M x M x 3 kernel `patch`.
# Returns an H x W matrix of inner products with zero-padding so the output
# has the same spatial dimensions as the input. (The notebook uses padding=0
# in F.conv2d, producing an (H-M+1)x(W-M+1) map; we use 'same' padding here
# purely so the displayed map aligns visually with the input image. The pooled
# feature is barely affected by this choice for typical M.)
convolve_patch <- function(img, patch) {
  H <- dim(img)[1]; W <- dim(img)[2]
  M <- dim(patch)[1]
  half <- (M - 1) %/% 2

  P  <- array(0, dim = c(H + M - 1, W + M - 1, 3))
  P[(half + 1):(half + H), (half + 1):(half + W), ] <- img

  out <- matrix(0, H, W)
  for (di in 0:(M - 1)) {
    for (dj in 0:(M - 1)) {
      for (ch in 1:3) {
        w <- patch[di + 1, dj + 1, ch]
        if (w == 0) next
        out <- out + w * P[(di + 1):(di + H),
                           (dj + 1):(dj + W), ch]
      }
    }
  }
  out
}

# Extract an M x M x 3 patch centered at (row, col), clipping to image bounds.
extract_patch <- function(img, row, col, M) {
  H <- dim(img)[1]; W <- dim(img)[2]
  half <- (M - 1) %/% 2

  r0 <- row - half;    c0 <- col - half
  r1 <- r0 + M - 1;    c1 <- c0 + M - 1

  if (r0 < 1)  { r0 <- 1;     r1 <- M }
  if (c0 < 1)  { c0 <- 1;     c1 <- M }
  if (r1 > H)  { r1 <- H;     r0 <- H - M + 1 }
  if (c1 > W)  { c1 <- W;     c0 <- W - M + 1 }

  list(patch = img[r0:r1, c0:c1, , drop = FALSE],
       bbox  = c(r0 = r0, c0 = c0, r1 = r1, c1 = c1))
}

# Rescale a patch to [0,1] for display (whitened patches have arbitrary range).
rescale_for_display <- function(x) {
  mn <- min(x); mx <- max(x)
  if (mx - mn < 1e-12) return(array(0.5, dim = dim(x)))
  (x - mn) / (mx - mn)
}

# ---- UI ----------------------------------------------------------------------

ui <- fluidPage(
  tags$head(tags$style(HTML("
    .well { background: #fafafa; }
    .top-controls {
      background: #fafafa;
      border: 1px solid #e5e5e5;
      border-radius: 6px;
      padding: 14px 16px;
    }
    .selection-row {
      padding: 12px 8px;
      margin: 10px 0;
      border-radius: 6px;
    }
    .feature-value {
      font-family: 'Courier New', monospace;
      font-size: 22px;
      font-weight: 700;
      text-align: center;
      margin-top: 30px;
    }
    .feature-caption {
      color: #777;
      text-align: center;
      font-size: 11px;
    }
    .panel-title {
      font-weight: 600;
      font-size: 12px;
      margin-bottom: 4px;
      text-align: center;
      color: #555;
    }
    .image-wrap {
      display: flex;
      justify-content: center;
      align-items: center;
    }
  "))),

  titlePanel("MOSAIKS Random Convolutional Features Visualizer"),
  tags$p(
    "Click three points on the image to define empirical patches ",
    HTML("(P<sub>1</sub>, P<sub>2</sub>, P<sub>3</sub>)"),
    " or generate them with a seed. Adjust the kernel size and bias, then ",
    tags$b("Submit"),
    " to whiten each patch (zero-mean / L2 / ZCA), compute the convolution ",
    "map and ReLU activation map, and pool the feature value."
  ),

  # ===========================================================================
  # TOP: controls (left) + input image (right). Constrained so the page below
  # uses full width.
  # ===========================================================================
  fluidRow(
    column(
      width = 4,
      div(
        class = "top-controls",
        selectInput("image_choice", "Image:", choices = NULL, selected = "miami.jpg"),

        tags$strong("Patch selection"),
        radioButtons(
          "selection_mode", NULL,
          choices = c("Random (with seed)" = "random",
          "Click on image" = "click"),
          selected = "random", inline = FALSE
        ),
        conditionalPanel(
          "input.selection_mode == 'random'",
          numericInput("seed", "Seed:", value = 17, min = 0, step = 1)
        ),
        conditionalPanel(
          "input.selection_mode == 'click'",
          helpText("Click 3 times on the image."),
          actionButton("clear_clicks", "Clear clicks",
                       class = "btn-sm btn-default")
        ),
        htmlOutput("click_status"),

        tags$hr(),
        fluidRow(
          column(6,
            sliderInput("kernel_size", "Kernel size (M):",
                        min = 3, max = 9, value = 3, step = 1)
          ),
          column(6,
            sliderInput("bias", "Bias (b):",
                        min = -2, max = 2, value = 0, step = 0.1)
          )
        ),
        selectInput("palette", "Heatmap palette:",
                    choices = names(PALETTES), selected = "Viridis"),
        fluidRow(
          column(2,
            
        radioButtons(
          "scales", "Scales:",
          choices = c("Free" = "free", "Fixed" = "fixed"),
          selected = "free", inline = FALSE
        ),
          ),
          column(10,
            
        helpText(br(),"Idependent color scales. \n",br(), 
          "Shared color scale across conv. and act. maps respectively."
        #)
        ),
          )
        ),
        sliderInput("compute_max", "Compute resolution (long edge):",
                    min = COMPUTE_MAX_RANGE[1], max = COMPUTE_MAX_RANGE[2],
                    value = COMPUTE_MAX_DEFAULT, step = 32),
        helpText(
          "Higher = more detail in the conv / activation maps, but ",
          "slower. Source image is always shown at full resolution."
        ),

        actionButton("submit", "Submit",
                     class = "btn-primary btn-block",
                     icon = icon("play"))
      )
    ),
    column(
      width = 8,
      div(class = "image-wrap",
        tags$div(
          style = "max-width: 720px; width: 100%;",
          tags$div(
            style = "color:#666; font-size:13px; margin-bottom:6px;
                     text-align: center;",
            "Click anywhere on the image to drop a patch. Boxes show the ",
            "kernel footprint at the current size."
          ),
          plotOutput("source_plot", click = "src_click",
                     width  = "100%", height = "640px")
        )
      )
    )
  ),

  tags$hr(),

  # ===========================================================================
  # BOTTOM: full-width results, one row per selection.
  # ===========================================================================
  tags$h4("Featurization results"),
  uiOutput("results_ui")
)

# ---- Server ------------------------------------------------------------------

server <- function(input, output, session) {

  # Discover available images on startup
  available_images <- reactive({
    files <- Sys.glob(file.path(IMAGE_DIR, IMAGE_GLOB))
    if (length(files) == 0) {
      files <- list.files(
        IMAGE_DIR,
        pattern = "\\.(png|jpe?g)$", ignore.case = TRUE, full.names = TRUE
      )
    }
    files
  })

  observe({
    files <- available_images()
    if (length(files) == 0) {
      updateSelectInput(session, "image_choice",
                        choices = c("(no images found)" = ""))
    } else {
      names(files) <- basename(files)
      default <- files[basename(files) == "miami.jpg"]
      selected <- if (length(default) > 0) unname(default[1]) else files[1]
      updateSelectInput(session, "image_choice",
                        choices = files, selected = selected)
    }
  })

  # Read the raw image once per file selection (cached). All other reactives
  # derive from this so we don't re-decode on every slider tick.
  raw_img <- reactive({
    req(input$image_choice, nzchar(input$image_choice))
    read_image(input$image_choice)
  })

  # Display version: high resolution, capped at DISPLAY_MAX. This is what the
  # user sees and clicks on. Independent of the compute slider so changing
  # compute resolution doesn't re-render or jitter the displayed image.
  display_img <- reactive({
    downsample(raw_img(), DISPLAY_MAX)
  })

  # Compute version: lower resolution, controlled by the compute_max slider.
  # `convolve_patch` runs on this. Patches are also extracted from this so the
  # kernel and the conv map share a coordinate system.
  compute_img <- reactive({
    downsample(raw_img(), input$compute_max)
  })

  # Stored click coordinates, in COMPUTE-image space (rows/cols of compute_img).
  # Storing in compute coords means downstream code is dimension-clean and
  # patches always extract from the same image they were clicked on.
  clicks <- reactiveVal(list())

  observeEvent(input$selection_mode, { clicks(list()) })
  observeEvent(input$image_choice,  { clicks(list()) })
  observeEvent(input$clear_clicks,  { clicks(list()) })
  # Changing compute resolution invalidates prior clicks (their pixel meaning
  # depends on the resolution they were captured at).
  observeEvent(input$compute_max,   { clicks(list()) })

  # Capture clicks on the source image. The plot's user coordinates are set
  # to the DISPLAY image's pixel grid (see source_plot below), so we map
  # display coords -> compute coords by ratio.
  observeEvent(input$src_click, {
    req(input$selection_mode == "click")
    disp <- display_img()
    comp <- compute_img()
    H_d <- dim(disp)[1]; W_d <- dim(disp)[2]
    H_c <- dim(comp)[1]; W_c <- dim(comp)[2]

    x <- input$src_click$x
    y <- input$src_click$y
    if (is.null(x) || is.null(y)) return()

    # Display-space pixel (y is plotted bottom-up, flip it back to row-down)
    col_d <- max(1, min(W_d, round(x)))
    row_d <- max(1, min(H_d, round(H_d - y + 1)))

    # Map to compute-space pixel
    col_c <- max(1, min(W_c, round(col_d * W_c / W_d)))
    row_c <- max(1, min(H_c, round(row_d * H_c / H_d)))

    cur <- clicks()
    if (length(cur) >= 3) cur <- list()
    cur[[length(cur) + 1]] <- c(row = row_c, col = col_c)
    clicks(cur)
  })

  random_points <- reactive({
    req(input$selection_mode == "random")
    img <- compute_img()
    H <- dim(img)[1]; W <- dim(img)[2]
    M <- input$kernel_size
    half <- (M - 1) %/% 2
    set.seed(input$seed)
    rows <- sample((half + 1):(H - half), 3, replace = TRUE)
    cols <- sample((half + 1):(W - half), 3, replace = TRUE)
    Map(function(r, c) c(row = r, col = c), rows, cols)
  })

  active_points <- reactive({
    if (input$selection_mode == "random") random_points() else clicks()
  })

  output$click_status <- renderUI({
    pts <- active_points()
    if (length(pts) == 0) {
      return(tags$div(style = "color:#888; margin-top:6px;",
                      "No points selected yet."))
    }
    rows <- lapply(seq_along(pts), function(i) {
      pt <- pts[[i]]
      tags$div(
        style = sprintf(
          "margin-top:4px; font-family:monospace; font-size:12px;
           color:%s; font-weight:600;", SEL_COLORS[i]),
        sprintf("\u25CF  P%d  row=%d  col=%d",
                i, pt[["row"]], pt[["col"]])
      )
    })
    do.call(tagList, rows)
  })

  # Plot the source image (HIGH-RES display version) with selection boxes.
  # Active_points() gives compute-space coords; we map them to display coords
  # so the boxes line up with the user's clicks at any compute resolution.
  output$source_plot <- renderPlot({
    disp <- display_img()
    comp <- compute_img()
    H_d <- dim(disp)[1]; W_d <- dim(disp)[2]
    H_c <- dim(comp)[1]; W_c <- dim(comp)[2]
    sx <- W_d / W_c    # compute -> display scale (x)
    sy <- H_d / H_c    # compute -> display scale (y)

    par(mar = c(0, 0, 0, 0), bg = "white")
    plot(c(0, W_d), c(0, H_d), type = "n", asp = 1, xlab = "", ylab = "",
         xaxs = "i", yaxs = "i", axes = FALSE)
    rasterImage(disp, 0, 0, W_d, H_d, interpolate = TRUE)

    pts <- active_points()
    M   <- input$kernel_size

    for (i in seq_along(pts)) {
      pt  <- pts[[i]]
      # extract_patch only needs bbox in compute coords; we then scale to display
      ext <- extract_patch(comp, pt[["row"]], pt[["col"]], M)
      bb  <- ext$bbox
      # Map bbox (compute pixels) -> display plot coords (y is bottom-up)
      x_left   <- (bb[["c0"]] - 1) * sx
      x_right  <-  bb[["c1"]]      * sx
      y_bottom <- H_d - bb[["r1"]] * sy
      y_top    <- H_d - (bb[["r0"]] - 1) * sy

      rect(x_left - 1, y_bottom - 1, x_right + 1, y_top + 1,
           border = "white", lwd = 5)
      rect(x_left, y_bottom, x_right, y_top,
           border = SEL_COLORS[i], lwd = 3.5)

      cx <- (x_left + x_right) / 2
      cy <- (y_bottom + y_top) / 2
      r  <- max(M * 1.4 * mean(c(sx, sy)), 8)
      symbols(cx, cy, circles = r, inches = FALSE,
              fg = SEL_COLORS[i], bg = NA, lwd = 4, add = TRUE)
      symbols(cx, cy, circles = r, inches = FALSE,
              fg = "white", bg = NA, lwd = 1.5, add = TRUE)

      # Anchor the index label at a fixed offset ABOVE the geometry so it
      # stays readable when the kernel (and therefore the box/circle) is
      # small. Use the taller of the box-half-height and the circle radius
      # as the anchor, then add a constant pixel-space gap on top.
      label_cex <- 1.6
      box_half_h <- (y_top - y_bottom) / 2
      anchor_top <- cy + max(box_half_h, r)
      # Convert a pixel-space gap and text height into user coords using
      # par("cxy") (size of one character in user units at cex = 1).
      ch <- par("cxy")[2]
      gap <- ch * 0.4
      label_y <- anchor_top + gap + ch * label_cex / 2

      # White halo for legibility against any underlying imagery
      for (dx in c(-1, 0, 1)) for (dy in c(-1, 0, 1)) {
        if (dx == 0 && dy == 0) next
        text(cx + dx * ch * 0.06, label_y + dy * ch * 0.06,
             labels = i, col = "white", font = 2, cex = label_cex)
      }
      text(cx, label_y, labels = i,
           col = SEL_COLORS[i], font = 2, cex = label_cex)
    }
  })

  # ---- Computation triggered by Submit ---------------------------------------

  results <- eventReactive(input$submit, {
    img <- compute_img()
    pts <- active_points()
    validate(
      need(length(pts) == 3,
           "Please select exactly 3 points (or use seeded random mode).")
    )
    M    <- input$kernel_size
    bias <- input$bias

    withProgress(message = "Computing...", value = 0, {
      # 1. Extract the 3 raw patches
      incProgress(0.1, detail = "Extracting patches")
      ext_list <- lapply(pts, function(pt) {
        extract_patch(img, pt[["row"]], pt[["col"]], M)
      })
      raw_patches <- array(0, dim = c(3, M, M, 3))
      for (i in 1:3) raw_patches[i, , , ] <- ext_list[[i]]$patch

      # 2. Whiten the stack of 3 patches together (zero-mean / L2 / ZCA),
      #    matching the notebook's `_normalize` step.
      incProgress(0.2, detail = "Whitening patches")
      whitened <- whiten_patches(raw_patches)

      # 3. For each patch: convolve with whitened kernel, add bias, ReLU, pool.
      lapply(seq_along(pts), function(i) {
        incProgress(0.6 / length(pts),
                    detail = sprintf("Convolving patch %d / %d",
                                     i, length(pts)))
        kernel <- whitened[i, , , , drop = FALSE]
        dim(kernel) <- c(M, M, 3)
        conv <- convolve_patch(img, kernel)
        # conv_with_bias is what F.conv2d(..., bias=biases) returns in PyTorch
        conv_with_bias <- conv + bias
        act <- pmax(conv_with_bias, 0)
        list(
          point          = pts[[i]],
          bbox           = ext_list[[i]]$bbox,
          patch_raw      = ext_list[[i]]$patch,    # M x M x 3, in [0,1]
          patch_whitened = kernel,                 # M x M x 3, arbitrary range
          conv           = conv_with_bias,         # H x W (post-bias, pre-ReLU)
          act            = act,                    # H x W
          feature        = mean(act)               # global average pool
        )
      })
    })
  })

  # Shared min/max across all three panels, used when input$scales == "fixed".
  # Computed once per `results()` so all three renderPlots see the same range.
  shared_ranges <- reactive({
    res <- results()
    conv_vals <- unlist(lapply(res, function(r) r$conv))
    act_vals  <- unlist(lapply(res, function(r) r$act))
    list(
      conv = c(min(conv_vals), max(conv_vals)),
      act  = c(min(act_vals),  max(act_vals))
    )
  })

  # Currently selected color palette (function n -> n colors)
  palette_fn <- reactive({
    PALETTES[[input$palette %||% "Viridis"]]
  })

  # Build one row of plots for a single selection.
  # Layout (12 cols total): patch (2) | conv (5) | act (4) | feat (1).
  # The whitened patch (the actual conv kernel after ZCA) is computed
  # under the hood -- if you want to expose it visually, add another
  # column here that calls plotOutput("white_<idx>"); the renderPlot is
  # still wired up below.
  render_row <- function(idx) {
    img <- compute_img()
    H   <- dim(img)[1]; W <- dim(img)[2]
    res <- results()[[idx]]
    color <- SEL_COLORS[idx]

    fluidRow(
      class = "selection-row",
      style = sprintf(
        "border-left: 6px solid %s; background: %s11;", color, color),

      column(2,
        tags$div(class = "panel-title",
                 style = sprintf("color:%s;", color),
                 sprintf("Patch %d", idx)),
        plotOutput(sprintf("patch_%d", idx), height = "260px")
      ),
      column(5,
        tags$div(class = "panel-title", "Convolution map (W * I + b)"),
        plotOutput(sprintf("conv_%d", idx), height = "260px")
      ),
      column(4,
        tags$div(class = "panel-title", "Activation map (ReLU)"),
        plotOutput(sprintf("act_%d", idx), height = "260px")
      ),
      column(1,
        tags$div(class = "panel-title", "Feature"),
        tags$div(
          class = "feature-value",
          style = sprintf("color:%s; font-size:16px; margin-top:80px;",
                          color),
          sprintf("x_%d =", idx)
        ),
        tags$div(
          class = "feature-value",
          style = sprintf("color:%s; margin-top:4px; font-size:18px;",
                          color),
          sprintf("%.4f", res$feature)
        ),
        tags$div(
          class = "feature-caption",
          sprintf("mean of %d\u00d7%d", H, W)
        )
      )
    )
  }

  output$results_ui <- renderUI({
    req(results())
    do.call(tagList, lapply(seq_along(results()), render_row))
  })

  # Per-row plots. Outputs registered lazily based on results().
  observe({
    res <- results()
    pal <- palette_fn()
    for (i in seq_along(res)) {
      local({
        ii  <- i
        clr <- SEL_COLORS[ii]

        # Raw patch thumbnail (the M x M x 3 sub-image). Interpolated so the
        # tiny kernel doesn't look harshly pixelated when blown up to ~250px.
        output[[sprintf("patch_%d", ii)]] <- renderPlot({
          patch <- res[[ii]]$patch_raw
          M <- dim(patch)[1]
          par(mar = c(0, 0, 0, 0), bg = "white")
          plot(c(0, M), c(0, M), type = "n", asp = 1, xlab = "", ylab = "",
               xaxs = "i", yaxs = "i", axes = FALSE)
          rasterImage(patch, 0, 0, M, M, interpolate = FALSE)
          rect(0, 0, M, M, border = clr, lwd = 4)
        }, res = 96)

        # Whitened patch (the actual conv kernel). Whitened values are
        # signed and unbounded, so we min-max stretch each channel block to
        # [0,1] before showing as RGB -- matches the notebook's display step.
        output[[sprintf("white_%d", ii)]] <- renderPlot({
          patch <- res[[ii]]$patch_whitened
          M <- dim(patch)[1]
          disp <- rescale_for_display(patch)
          par(mar = c(0, 0, 0, 0), bg = "white")
          plot(c(0, M), c(0, M), type = "n", asp = 1, xlab = "", ylab = "",
               xaxs = "i", yaxs = "i", axes = FALSE)
          rasterImage(disp, 0, 0, M, M, interpolate = TRUE)
          rect(0, 0, M, M, border = clr, lwd = 4, lty = 2)
        }, res = 96)

        # Convolution map (post-bias, pre-ReLU). Drawn via rasterImage on an
        # asp=1 canvas so the map keeps the input image's aspect ratio
        # (image() would stretch to fill the plot box).
        output[[sprintf("conv_%d", ii)]] <- renderPlot({
          conv <- res[[ii]]$conv
          H <- nrow(conv); W <- ncol(conv)
          if (isTRUE(input$scales == "fixed")) {
            rng <- shared_ranges()$conv
            mn <- rng[1]; mx <- rng[2]
          } else {
            mn <- min(conv); mx <- max(conv)
          }
          if (mx - mn < 1e-12) { mn <- mn - 1; mx <- mx + 1 }
          scaled <- (conv - mn) / (mx - mn)
          # Map scaled values [0,1] -> palette colors -> H x W color matrix
          color_idx <- pmin(256, pmax(1, round(scaled * 255) + 1))
          color_mat <- matrix(pal(256)[color_idx], nrow = H, ncol = W)
          par(mar = c(0, 0, 0, 0), bg = "white")
          plot(c(0, W), c(0, H), type = "n", asp = 1, xlab = "", ylab = "",
               xaxs = "i", yaxs = "i", axes = FALSE)
          rasterImage(color_mat, 0, 0, W, H, interpolate = TRUE)
        }, res = 96)

        # Activation map (ReLU of conv) -- same aspect-preserving treatment.
        output[[sprintf("act_%d", ii)]] <- renderPlot({
          act <- res[[ii]]$act
          H <- nrow(act); W <- ncol(act)
          if (isTRUE(input$scales == "fixed")) {
            rng <- shared_ranges()$act
            mn <- rng[1]; mx <- rng[2]
          } else {
            mn <- min(act); mx <- max(act)
          }
          if (mx - mn < 1e-12) { mn <- mn - 1; mx <- mx + 1 }
          scaled <- (act - mn) / (mx - mn)
          color_idx <- pmin(256, pmax(1, round(scaled * 255) + 1))
          color_mat <- matrix(pal(256)[color_idx], nrow = H, ncol = W)
          par(mar = c(0, 0, 0, 0), bg = "white")
          plot(c(0, W), c(0, H), type = "n", asp = 1, xlab = "", ylab = "",
               xaxs = "i", yaxs = "i", axes = FALSE)
          rasterImage(color_mat, 0, 0, W, H, interpolate = TRUE)
        }, res = 96)
      })
    }
  })
}

# Tiny helper for `input$palette %||% "Magma"` style fallback
`%||%` <- function(a, b) if (is.null(a) || !nzchar(a)) b else a

# ---- Run ---------------------------------------------------------------------
shinyApp(ui, server)
